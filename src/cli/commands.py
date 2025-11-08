"""
命令行界面命令实现
"""

import os

from ..managers.dataset_manager import UnifiedDatasetManager
from ..managers.config_manager import get_config_manager


def process_datasets(dataset_names=None, skip_images=False, max_workers=8):
    """处理数据集"""
    print("=== 统一数据集处理框架 ===")

    # 获取配置管理器
    config_manager = get_config_manager()

    # 创建统一管理器
    manager = UnifiedDatasetManager(max_workers=max_workers)

    # 确定要处理的数据集
    if dataset_names is None:
        # 处理所有数据集
        configs = config_manager.get_all_configs()
        dataset_names = [config.name for config in configs]
    else:
        # 检查指定的数据集是否存在
        available_datasets = config_manager.list_datasets()
        invalid_datasets = [
            name for name in dataset_names if name not in available_datasets
        ]
        if invalid_datasets:
            print(f"错误: 以下数据集不存在: {', '.join(invalid_datasets)}")
            print(f"可用的数据集: {', '.join(available_datasets)}")
            return False

        configs = [config_manager.get_config(name) for name in dataset_names]

    # 注册数据集
    print("\n注册数据集...")
    for config in configs:
        try:
            manager.register_dataset(config)
        except Exception as e:
            print(f"注册数据集 {config.name} 失败: {e}")
            continue

    # 处理数据集
    print("\n开始处理数据集...")
    results = {}
    for dataset_name in dataset_names:
        if dataset_name in manager.processors:
            try:
                print(f"\n处理 {dataset_name}...")
                result = manager.process_dataset(dataset_name)
                if result[0] is not None:
                    results[dataset_name] = result
                    print(f"✓ {dataset_name} 处理成功")
                else:
                    print(f"✗ {dataset_name} 处理失败")
            except Exception as e:
                print(f"✗ {dataset_name} 处理时出现异常: {e}")

    # 显示总体统计
    if results:
        print("\n=== 处理结果汇总 ===")
        total_ratings = 0
        total_users = 0
        total_items = 0

        for name, (ratings_df, user_df, item_df) in results.items():
            print(f"{name}:")
            print(f"  评分: {len(ratings_df):,}")
            print(f"  用户: {len(user_df):,}")
            print(f"  商品: {len(item_df):,}")

            total_ratings += len(ratings_df)
            total_users += len(user_df)
            total_items += len(item_df)

        print(f"\n总计:")
        print(f"  评分: {total_ratings:,}")
        print(f"  用户: {total_users:,}")
        print(f"  商品: {total_items:,}")

    # 下载图片（如果需要）
    if not skip_images and results:
        print("\n开始下载图片...")
        image_results = {}
        for dataset_name in results.keys():
            try:
                print(f"\n下载 {dataset_name} 图片...")
                stats = manager.download_images(dataset_name)
                image_results[dataset_name] = stats
            except Exception as e:
                print(f"下载 {dataset_name} 图片时出现异常: {e}")

        # 显示图片下载统计
        if image_results:
            print("\n=== 图片下载结果汇总 ===")
            for name, stats in image_results.items():
                if stats:
                    success_rate = 0
                    if stats["downloaded"] + stats["failed"] > 0:
                        success_rate = (
                            stats["downloaded"]
                            / (stats["downloaded"] + stats["failed"])
                            * 100
                        )
                    print(f"{name}:")
                    print(f"  已存在: {stats['existing']}")
                    print(f"  成功下载: {stats['downloaded']}")
                    print(f"  下载失败: {stats['failed']}")
                    print(f"  无效URL: {stats['invalid_url']}")
                    print(f"  成功率: {success_rate:.1f}%")

    print("\n=== 处理完成 ===")
    return True


def list_datasets():
    """列出所有可用的数据集"""
    config_manager = get_config_manager()
    datasets = config_manager.list_datasets()

    print("=== 可用数据集 ===")
    for dataset_name in datasets:
        config = config_manager.get_config(dataset_name)
        print(f"{dataset_name} ({config.dataset_type})")

        # 检查输入文件是否存在
        missing_files = []
        for file_type, file_path in config.input_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")

        if missing_files:
            print(f"  ⚠️  缺失文件: {', '.join(missing_files)}")
        else:
            print(f"  ✓ 所有输入文件存在")

    print(f"\n总共 {len(datasets)} 个数据集")


def show_config(dataset_name):
    """显示指定数据集的配置"""
    config_manager = get_config_manager()
    config = config_manager.get_config(dataset_name)

    if config is None:
        print(f"错误: 数据集 '{dataset_name}' 不存在")
        available_datasets = config_manager.list_datasets()
        print(f"可用的数据集: {', '.join(available_datasets)}")
        return

    print(f"=== {dataset_name} 配置信息 ===")
    print(f"名称: {config.name}")
    print(f"类型: {config.dataset_type}")
    print(f"输出目录: {config.output_dir}")
    print("输入文件:")
    for file_type, file_path in config.input_files.items():
        exists = "✓" if os.path.exists(file_path) else "✗"
        print(f"  {file_type}: {file_path} {exists}")
    print("列映射:")
    for logical_name, actual_name in config.columns_mapping.items():
        print(f"  {logical_name} -> {actual_name}")