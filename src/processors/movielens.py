"""
MovieLens数据集处理器
"""

import pandas as pd
import os
from typing import Dict, Any

from .base import DatasetProcessor
from ..utils.logging_config import logger


class MovieLensDatasetProcessor(DatasetProcessor):
    """MovieLens数据集处理器"""

    def load_ratings_data(self) -> pd.DataFrame:
        """加载MovieLens评分数据"""
        ratings_file = self.config.input_files.get("ratings")
        if not ratings_file or not os.path.exists(ratings_file):
            logger.error(f"评分文件不存在: {ratings_file}")
            return None

        try:
            df = pd.read_csv(ratings_file)
            logger.info(f"成功加载 {len(df)} 条评分记录")

            # MovieLens数据已经很干净，只需要基本检查
            required_cols = ["userId", "movieId", "rating", "timestamp"]
            df = df.dropna(subset=required_cols)

            logger.info(f"清理后剩余 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载评分数据失败: {e}")
            return None

    def load_metadata(self) -> Dict[str, Any]:
        """加载MovieLens元数据"""
        movies_file = self.config.input_files.get("movies")
        links_file = self.config.input_files.get("links")

        metadata = {}

        # 加载电影信息
        if movies_file and os.path.exists(movies_file):
            try:
                movies_df = pd.read_csv(movies_file)
                for _, row in movies_df.iterrows():
                    metadata[row["movieId"]] = {
                        "title": row["title"],
                        "genres": row["genres"],
                        "imdb_id": "",
                        "tmdb_id": "",
                    }
                logger.info(f"成功加载 {len(movies_df)} 部电影信息")
            except Exception as e:
                logger.error(f"加载电影信息失败: {e}")

        # 加载链接信息
        if links_file and os.path.exists(links_file):
            try:
                links_df = pd.read_csv(links_file)
                for _, row in links_df.iterrows():
                    movie_id = row["movieId"]
                    if movie_id in metadata:
                        metadata[movie_id]["imdb_id"] = str(row.get("imdbId", ""))
                        metadata[movie_id]["tmdb_id"] = str(row.get("tmdbId", ""))
                logger.info(f"成功加载 {len(links_df)} 个电影链接")
            except Exception as e:
                logger.error(f"加载链接信息失败: {e}")

        return metadata

    def extract_image_url(self, metadata_item: Any) -> str:
        """从MovieLens元数据中提取图片URL"""
        # MovieLens数据集通常不包含直接的图片URL
        # 这里可以通过IMDB ID或TMDB ID来获取图片，但需要额外的API调用
        # 实际使用时可以通过IMDB ID或TMDB ID来获取图片
        return ""