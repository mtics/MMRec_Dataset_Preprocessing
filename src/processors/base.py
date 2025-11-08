"""
数据集处理器抽象基类
"""

import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

from ..models.dataset_config import DatasetConfig
from ..utils.logging_config import logger


class DatasetProcessor(ABC):
    """数据集处理器抽象基类"""

    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load_ratings_data(self) -> pd.DataFrame:
        """加载评分数据"""
        pass

    @abstractmethod
    def load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        pass

    @abstractmethod
    def extract_image_url(self, metadata_item: Any) -> str:
        """从元数据中提取图片URL"""
        pass

    def create_id_mapping(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """创建ID映射"""
        logger.info("创建ID映射...")
        
        # 提取列名
        user_col = self.config.columns_mapping["user_id"]
        item_col = self.config.columns_mapping["item_id"]
        
        # 创建用户ID映射
        unique_users = df[user_col].unique()
        user_pairs = pd.DataFrame({
            "originalUserID": unique_users,
            "userID": range(1, len(unique_users) + 1)
        })
        
        # 创建商品ID映射
        unique_items = df[item_col].unique()
        item_pairs = pd.DataFrame({
            "originalItemID": unique_items,
            "itemID": range(1, len(unique_items) + 1)
        })
        
        logger.info(f"创建映射完成 - 用户: {len(user_pairs)}, 商品: {len(item_pairs)}")
        return user_pairs, item_pairs

    def process_dataset(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """处理数据集的主要流程"""
        logger.info(f"开始处理数据集: {self.config.name}")
        
        # 加载评分数据
        ratings_df = self.load_ratings_data()
        if ratings_df is None:
            return None, None, None
            
        # 加载元数据
        metadata = self.load_metadata()
        
        # 创建ID映射
        user_pairs_df, item_pairs_df = self.create_id_mapping(ratings_df)
        
        # 应用映射到评分数据
        user_col = self.config.columns_mapping["user_id"]
        item_col = self.config.columns_mapping["item_id"]
        rating_col = self.config.columns_mapping["rating"]
        timestamp_col = self.config.columns_mapping["timestamp"]
        
        # 创建映射字典
        user_mapping = dict(zip(user_pairs_df["originalUserID"], user_pairs_df["userID"]))
        item_mapping = dict(zip(item_pairs_df["originalItemID"], item_pairs_df["itemID"]))
        
        # 应用映射
        ratings_df["userID"] = ratings_df[user_col].map(user_mapping)
        ratings_df["itemID"] = ratings_df[item_col].map(item_mapping)
        ratings_df["rating"] = ratings_df[rating_col]
        ratings_df["timestamp"] = ratings_df[timestamp_col]
        
        # 只保留需要的列
        final_ratings = ratings_df[["userID", "itemID", "rating", "timestamp"]].copy()
        
        # 为item_pairs添加元数据 - 使用向量化操作优化性能
        logger.info("添加商品元数据...")
        
        # 创建元数据映射字典
        title_mapping = {}
        url_mapping = {}
        
        for original_id, item_info in metadata.items():
            title_mapping[original_id] = item_info.get("title", "")
            url_mapping[original_id] = self.extract_image_url(item_info)
        
        # 使用向量化操作映射元数据
        item_pairs_df["title"] = item_pairs_df["originalItemID"].map(title_mapping).fillna("")
        item_pairs_df["img_url"] = item_pairs_df["originalItemID"].map(url_mapping).fillna("")
        
        # 保存结果
        self.save_results(final_ratings, user_pairs_df, item_pairs_df)
        
        # 打印统计信息
        self.print_statistics(final_ratings, user_pairs_df, item_pairs_df)
        
        logger.info(f"{self.config.name} 数据集处理完成")
        return final_ratings, user_pairs_df, item_pairs_df

    def save_results(self, ratings_df: pd.DataFrame, user_pairs_df: pd.DataFrame, item_pairs_df: pd.DataFrame):
        """保存处理结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存文件
        ratings_file = os.path.join(self.config.output_dir, "ratings.csv")
        user_pairs_file = os.path.join(self.config.output_dir, "user_pairs.csv")
        item_pairs_file = os.path.join(self.config.output_dir, "item_pairs.csv")
        dataset_item_file = os.path.join(self.config.output_dir, f"{self.config.name}_item.csv")
        
        logger.info("保存处理结果...")
        
        # 对于大文件，使用优化的保存方式
        if len(ratings_df) > 1000000:  # 超过100万条记录
            logger.info(f"保存大文件({len(ratings_df):,}条记录)，使用优化模式...")
            ratings_df.to_csv(ratings_file, index=False, chunksize=100000)
        else:
            ratings_df.to_csv(ratings_file, index=False)
            
        user_pairs_df.to_csv(user_pairs_file, index=False)
        item_pairs_df.to_csv(item_pairs_file, index=False)
        
        # 创建简化的item文件
        simplified_items = item_pairs_df[["itemID", "title"]].copy()
        simplified_items.to_csv(dataset_item_file, index=False)
        
        logger.info(f"数据已保存到: {self.config.output_dir}")
        logger.info(f"简化item文件已保存: {dataset_item_file}")

    def print_statistics(self, ratings_df: pd.DataFrame, user_pairs_df: pd.DataFrame, item_pairs_df: pd.DataFrame):
        """打印统计信息"""
        print(f"\n{self.config.name} 数据集统计:")
        print(f"  评分记录: {len(ratings_df):,}")
        print(f"  用户数: {len(user_pairs_df):,}")
        print(f"  商品数: {len(item_pairs_df):,}")
        print(f"  评分范围: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")

        # 元数据匹配统计
        matched_items = (item_pairs_df["title"] != "").sum()
        matched_images = (item_pairs_df["img_url"] != "").sum()
        total_items = len(item_pairs_df)

        print(f"  标题匹配: {matched_items}/{total_items} ({matched_items/total_items*100:.1f}%)")
        print(f"  图片匹配: {matched_images}/{total_items} ({matched_images/total_items*100:.1f}%)")