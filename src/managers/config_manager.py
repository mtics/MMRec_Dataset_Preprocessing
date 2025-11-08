"""
数据集配置管理模块
集中管理不同推荐系统数据集的配置信息
"""

from typing import Dict, List, Optional, Any

from ..models.dataset_config import DatasetConfig


class DatasetConfigManager:
    """数据集配置管理器"""

    def __init__(self):
        self.configs = {}
        self._load_default_configs()

    def _load_default_configs(self):
        """加载默认配置"""
        # Amazon数据集配置
        self.add_amazon_config(
            name="Clothing",
            ratings_file="amazon/dataset/ratings_Clothing_Shoes_and_Jewelry.csv",
            metadata_file="amazon/dataset/meta_Clothing_Shoes_and_Jewelry.json",
            output_dir="processed/Clothing",
        )

        self.add_amazon_config(
            name="Sports",
            ratings_file="amazon/dataset/ratings_Sports_and_Outdoors.csv",
            metadata_file="amazon/dataset/meta_Sports_and_Outdoors.json",
            output_dir="processed/Sports",
        )
        
        self.add_amazon_config(
            name="Baby",
            ratings_file="amazon/dataset/ratings_Baby.csv",
            metadata_file="amazon/dataset/meta_Baby.json",
            output_dir="processed/Baby",
        )

        # MovieLens数据集配置
        self.add_movielens_config(
            name="ML",
            ratings_file="ml-latest-small/ratings.csv",
            movies_file="ml-latest-small/movies.csv",
            links_file="ml-latest-small/links.csv",
            output_dir="processed/MovieLens",
        )

    def add_amazon_config(
        self, name: str, ratings_file: str, metadata_file: str, output_dir: str
    ):
        """添加Amazon数据集配置"""
        config = DatasetConfig(
            name=name,
            dataset_type="amazon",
            input_files={"ratings": ratings_file, "metadata": metadata_file},
            output_dir=output_dir,
            columns_mapping={
                "user_id": "user_id",
                "item_id": "asin",
                "rating": "rating",
                "timestamp": "timestamp",
            },
        )
        self.configs[name] = config

    def add_movielens_config(
        self,
        name: str,
        ratings_file: str,
        movies_file: str,
        links_file: str,
        output_dir: str,
    ):
        """添加MovieLens数据集配置"""
        config = DatasetConfig(
            name=name,
            dataset_type="movielens",
            input_files={
                "ratings": ratings_file,
                "movies": movies_file,
                "links": links_file,
            },
            output_dir=output_dir,
            columns_mapping={
                "user_id": "userId",
                "item_id": "movieId",
                "rating": "rating",
                "timestamp": "timestamp",
            },
        )
        self.configs[name] = config

    def add_custom_config(self, config: DatasetConfig):
        """添加自定义配置"""
        self.configs[config.name] = config

    def get_config(self, name: str) -> Optional[DatasetConfig]:
        """获取配置"""
        return self.configs.get(name)

    def get_all_configs(self) -> List[DatasetConfig]:
        """获取所有配置"""
        return list(self.configs.values())

    def get_configs_by_type(self, dataset_type: str) -> List[DatasetConfig]:
        """根据类型获取配置"""
        return [
            config
            for config in self.configs.values()
            if config.dataset_type == dataset_type
        ]

    def list_datasets(self) -> List[str]:
        """列出所有数据集名称"""
        return list(self.configs.keys())

    def remove_config(self, name: str):
        """移除配置"""
        if name in self.configs:
            del self.configs[name]

    def update_config(self, name: str, **kwargs):
        """更新配置"""
        if name in self.configs:
            config = self.configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    def print_all_configs(self):
        """打印所有配置信息"""
        print("=== 数据集配置信息 ===")
        for name, config in self.configs.items():
            print(f"\n{name} ({config.dataset_type}):")
            print(f"  输入文件: {config.input_files}")
            print(f"  输出目录: {config.output_dir}")
            print(f"  列映射: {config.columns_mapping}")


# 全局配置管理器实例
config_manager = DatasetConfigManager()


def get_config_manager() -> DatasetConfigManager:
    """获取配置管理器实例"""
    return config_manager


if __name__ == "__main__":
    # 测试配置管理器
    manager = get_config_manager()
    manager.print_all_configs()