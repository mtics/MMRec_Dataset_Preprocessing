"""
数据集配置数据模型
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str  # 数据集名称
    dataset_type: str  # 数据集类型 (amazon, movielens等)
    input_files: Dict[str, str]  # 输入文件路径映射
    output_dir: str  # 输出目录
    columns_mapping: Dict[str, str]  # 列名映射
    custom_params: Dict[str, Any] = None  # 自定义参数

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}