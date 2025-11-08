"""
Amazon数据集处理器
"""

import pandas as pd
import ast
import os
from typing import Dict, Any
from tqdm import tqdm

from .base import DatasetProcessor
from ..utils.logging_config import logger


class AmazonDatasetProcessor(DatasetProcessor):
    """Amazon数据集处理器"""

    def load_ratings_data(self) -> pd.DataFrame:
        """加载Amazon评分数据"""
        ratings_file = self.config.input_files.get("ratings")
        if not ratings_file or not os.path.exists(ratings_file):
            logger.error(f"评分文件不存在: {ratings_file}")
            return None

        try:
            df = pd.read_csv(ratings_file)
            logger.info(f"成功加载 {len(df)} 条评分记录")

            # 处理Amazon特有的列名映射
            if "parent_asin" in df.columns and "asin" not in df.columns:
                df["asin"] = df["parent_asin"]

            # 数据清理
            required_cols = ["user_id", "asin", "rating", "timestamp"]
            df = df.dropna(subset=required_cols)

            logger.info(f"清理后剩余 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载评分数据失败: {e}")
            return None

    def load_metadata(self) -> Dict[str, Any]:
        """加载Amazon元数据"""
        meta_file = self.config.input_files.get("metadata")
        if not meta_file or not os.path.exists(meta_file):
            logger.warning(f"元数据文件不存在: {meta_file}")
            return {}

        metadata = {}
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                parse_errors = 0
                for line_num, line in enumerate(tqdm(f, desc="加载元数据"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # 尝试使用ast.literal_eval解析Python字典格式
                        data = ast.literal_eval(line)
                        asin = data.get("asin")
                        parent_asin = data.get("parent_asin")

                        # 提取图片URL，处理不同的字段名
                        images = data.get("images", [])
                        if not images and "imUrl" in data:
                            images = [data["imUrl"]]

                        item_info = {
                            "title": data.get("title", ""),
                            "images": images,
                        }

                        if asin:
                            metadata[asin] = item_info
                        if parent_asin and parent_asin != asin:
                            metadata[parent_asin] = item_info
                    except (ValueError, SyntaxError) as e:
                        parse_errors += 1
                        if parse_errors <= 5:  # 只记录前5个错误
                            logger.warning(f"解析第{line_num}行失败: {str(e)[:100]}")
                        continue
                    except Exception as e:
                        parse_errors += 1
                        if parse_errors <= 5:
                            logger.warning(f"处理第{line_num}行时出错: {str(e)[:100]}")
                        continue
                
                if parse_errors > 0:
                    logger.warning(f"共有 {parse_errors} 行解析失败")

            logger.info(f"成功加载 {len(metadata)} 个商品的元数据")
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")

        return metadata

    def extract_image_url(self, metadata_item: Any) -> str:
        """从Amazon元数据中提取图片URL"""
        images = metadata_item.get("images", [])
        if images and isinstance(images, list):
            for image in images:
                # 处理复杂字典格式（All_Beauty等）
                if isinstance(image, dict) and image.get("variant") == "MAIN":
                    large_img = image.get("large")
                    if large_img:
                        return large_img
                # 处理简单字符串格式（Baby/Clothing/Sports等）
                elif isinstance(image, str) and image.startswith("http"):
                    return image
            
            # 如果没有找到MAIN变体，尝试返回第一个有效的图片URL
            for image in images:
                if isinstance(image, dict):
                    large_img = image.get("large") or image.get("hi_res")
                    if large_img:
                        return large_img
                elif isinstance(image, str) and image.startswith("http"):
                    return image
        return ""