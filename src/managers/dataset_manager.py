"""
统一数据集管理器
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional

from ..models.dataset_config import DatasetConfig
from ..processors.amazon import AmazonDatasetProcessor
from ..processors.movielens import MovieLensDatasetProcessor
from ..downloaders.image_downloader import ImageDownloader
from ..utils.logging_config import logger


class UnifiedDatasetManager:
    """统一数据集管理器"""

    def __init__(self, max_workers: int = 6):
        self.processors = {}
        self.image_downloader = ImageDownloader(max_workers=max_workers)

    def register_dataset(self, config: DatasetConfig):
        """注册数据集配置"""
        if config.dataset_type == "amazon":
            processor = AmazonDatasetProcessor(config)
        elif config.dataset_type == "movielens":
            processor = MovieLensDatasetProcessor(config)
        else:
            raise ValueError(f"不支持的数据集类型: {config.dataset_type}")

        # 验证输入文件是否存在
        missing_files = []
        for file_type, file_path in config.input_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")

        if missing_files:
            raise FileNotFoundError(f"以下输入文件不存在: {', '.join(missing_files)}")

        self.processors[config.name] = processor
        logger.info(f"注册数据集: {config.name} ({config.dataset_type})")

    def process_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """处理指定的数据集"""
        if dataset_name not in self.processors:
            logger.error(f"数据集 {dataset_name} 未注册")
            return None, None, None

        processor = self.processors[dataset_name]
        return processor.process_dataset()

    def process_all_datasets(self) -> Dict[str, Tuple]:
        """处理所有已注册的数据集"""
        results = {}
        for name in self.processors:
            logger.info(f"处理数据集: {name}")
            result = self.process_dataset(name)
            results[name] = result
        return results

    def download_images(self, dataset_name: str) -> Dict[str, int]:
        """下载指定数据集的图片"""
        if dataset_name not in self.processors:
            logger.error(f"数据集 {dataset_name} 未注册")
            return {}

        processor = self.processors[dataset_name]
        item_pairs_file = os.path.join(processor.config.output_dir, "item_pairs.csv")
        cover_dir = os.path.join(processor.config.output_dir, "cover")

        return self.image_downloader.download_dataset_images(
            item_pairs_file, cover_dir, dataset_name
        )

    def download_all_images(self) -> Dict[str, Dict[str, int]]:
        """下载所有数据集的图片"""
        results = {}
        for name in self.processors:
            logger.info(f"下载数据集图片: {name}")
            stats = self.download_images(name)
            results[name] = stats
        return results