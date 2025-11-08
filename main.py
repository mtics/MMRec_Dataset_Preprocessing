#!/usr/bin/env python3
"""
统一数据集处理框架 - 主入口文件
支持多种推荐系统数据集的统一处理和映射
"""

import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.commands import process_datasets, list_datasets, show_config
from src.utils.logging_config import setup_logging


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="统一数据集处理框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            使用示例:
            # 处理所有数据集
            python main.py

            # 处理指定数据集
            python main.py --datasets All_Beauty Gift_Cards

            # 只处理数据，不下载图片
            python main.py --skip-images

            # 使用更多线程下载图片以加快速度
            python main.py --max-workers 16

            # 列出所有可用数据集
            python main.py --list

            # 显示指定数据集的配置
            python main.py --show-config All_Beauty
        """,
    )

    parser.add_argument("--datasets", nargs="+", help="指定要处理的数据集名称")

    parser.add_argument("--skip-images", action="store_true", help="跳过图片下载")

    parser.add_argument("--list", action="store_true", help="列出所有可用的数据集")

    parser.add_argument(
        "--show-config", metavar="DATASET", help="显示指定数据集的配置信息"
    )

    parser.add_argument(
        "--max-workers", type=int, default=8, metavar="N",
        help="图片下载的最大并发线程数 (默认: 8)"
    )

    args = parser.parse_args()

    # 处理不同的命令
    if args.list:
        list_datasets()
    elif args.show_config:
        show_config(args.show_config)
    else:
        # 处理数据集
        success = process_datasets(
            dataset_names=args.datasets, 
            skip_images=args.skip_images,
            max_workers=args.max_workers
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    # 设置日志
    setup_logging()
    
    # 运行主程序
    main()