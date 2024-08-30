import gzip
import pandas as pd
import os
import requests
import yaml


def parse(file_path):
    g = gzip.open(file_path, 'rb')
    for l in g:
        yield eval(l)


def getDF(file_path):
    if file_path.endswith('.gz'):
        i = 0
        data = {}
        for d in parse(file_path):
            data[i] = d
            i += 1
        return pd.DataFrame.from_dict(data, orient='index')
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path, sep=',', header=None, names=['asin', 'cn_title', 'title'])


def is_valid_url(url):
    # 简单检查URL格式是否有效
    if pd.isna(url) or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        return False
    return True


def download_images(data, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    invalid_ids = []  # 存储无效的imUrl对应的itemID

    # 遍历DataFrame中的每一行
    for index, row in data.iterrows():
        iid = row['itemID']
        image_url = row['imUrl']

        # 构造图片的保存路径
        file_path = os.path.join(target_folder, f"{iid}.jpg")

        # 检查 imUrl 是否有效
        if not is_valid_url(image_url):
            print(f"Invalid URL for itemID {iid}: {image_url}")
            invalid_ids.append(iid)
            continue  # 跳过此行，继续下一个

        # 检查图片文件是否已经存在
        if os.path.exists(file_path):
            print(f"Image already exists, skipping download: {file_path}")
            continue  # 如果文件已经存在，跳过下载

        try:
            # 下载图片
            response = requests.get(image_url)
            response.raise_for_status()  # 检查请求是否成功

            # 构造图片的保存路径
            file_path = os.path.join(target_folder, f"{iid}.jpg")

            # 将图片写入文件
            with open(file_path, 'wb') as file:
                file.write(response.content)

            print(f"Download image successfully: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error when downloading image: {image_url}, Error message: {e}")

    # 输出所有无效的imUrl对应的itemID
    if invalid_ids:
        print("Invalid imUrl found for the following itemIDs:")
        print(invalid_ids)
    else:
        print("All imUrls are valid.")


def read_yaml_to_dict(file_path):
    # 打开并读取 YAML 文件
    with open(file_path, 'r') as file:
        # 使用 yaml.safe_load() 将 YAML 内容加载为 Python 字典
        yaml_content = yaml.safe_load(file)
    return yaml_content


def compute_modalities_similarity(encoding1, encoding2):
    import numpy as np

    # 确保输入的两个编码具有相同的形状
    assert encoding1.shape == encoding2.shape, f"Two modality encodings must have the same shape. But got: {encoding1.shape} and {encoding2.shape}"

    # L2 归一化每个模态的特征向量
    encoding1_norm = encoding1 / np.linalg.norm(encoding1, axis=1, keepdims=True)
    encoding2_norm = encoding2 / np.linalg.norm(encoding2, axis=1, keepdims=True)

    # 计算每个 item 的两种模态之间的余弦相似度
    similarity = np.dot(encoding1_norm, encoding2_norm.T)
    item_similarity = np.diag(similarity)  # 对角线上的元素是每个 item 的相似度

    return item_similarity
