"""
高性能的多线程图片下载器
"""

import pandas as pd
import os
import requests
import time
import random
from urllib.parse import urlparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from typing import Dict, Tuple, Any

from ..utils.logging_config import logger


class ImageDownloader:
    """高性能的多线程图片下载器"""

    def __init__(self, timeout: int = 30, max_workers: int = 8, delay: float = 0.1):
        self.timeout = timeout
        self.max_workers = max_workers
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.amazon.com/",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-site",
            "DNT": "1",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }
        self.session_pool = {}
        self.lock = threading.Lock()

    def get_session(self) -> requests.Session:
        """获取线程本地的Session对象"""
        thread_id = threading.current_thread().ident
        with self.lock:
            if thread_id not in self.session_pool:
                session = requests.Session()
                
                # 配置重试策略 - 针对Amazon服务器的限制优化
                retry_strategy = Retry(
                    total=5,
                    backoff_factor=2.0,  # 指数退避
                    status_forcelist=[429, 500, 502, 503, 504, 403],  # 添加403禁止访问
                    connect=3,  # 连接重试次数
                    read=3,     # 读取重试次数
                )
                
                # 配置HTTP适配器
                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=10,
                    pool_maxsize=20
                )
                
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                session.headers.update(self.headers)
                
                self.session_pool[thread_id] = session
            
        return self.session_pool[thread_id]

    def get_image_extension(self, url: str, content_type: str = None) -> str:
        """根据URL或内容类型确定图片扩展名"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if path.endswith((".jpg", ".jpeg")):
            return ".jpg"
        elif path.endswith(".png"):
            return ".png"
        elif path.endswith(".gif"):
            return ".gif"
        elif path.endswith(".webp"):
            return ".webp"

        if content_type:
            content_type = content_type.lower()
            if "jpeg" in content_type or "jpg" in content_type:
                return ".jpg"
            elif "png" in content_type:
                return ".png"
            elif "gif" in content_type:
                return ".gif"
            elif "webp" in content_type:
                return ".webp"

        return ".jpg"

    def check_image_exists(self, item_id: str, cover_dir: str) -> Tuple[bool, str]:
        """检查图片是否已存在"""
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        for ext in extensions:
            file_path = os.path.join(cover_dir, f"{item_id}{ext}")
            if os.path.exists(file_path):
                return True, file_path
        return False, ""

    def download_single_image(self, url: str, save_path: str) -> Tuple[bool, str, str]:
        """下载单张图片"""
        if (
            pd.isna(url)
            or not url
            or str(url).strip() == ""
            or str(url).lower() == "nan"
        ):
            return False, "", "URL为空或无效"

        url = str(url).strip()
        session = self.get_session()

        # 添加较小的随机延时
        random_delay = random.uniform(0.01, 0.05)
        time.sleep(random_delay)

        try:
            response = session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            extension = self.get_image_extension(url, content_type)
            final_path = save_path + extension

            with open(final_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 验证图片
            try:
                with Image.open(final_path) as img:
                    img.verify()
            except Exception as e:
                if os.path.exists(final_path):
                    os.remove(final_path)
                return False, "", f"图片验证失败: {str(e)}"

            return True, final_path, "下载成功"

        except requests.exceptions.Timeout:
            return False, "", "请求超时"
        except requests.exceptions.ConnectionError:
            return False, "", "连接错误"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False, "", f"404 - 图片不存在"
            return False, "", f"HTTP错误: {e.response.status_code}"
        except Exception as e:
            return False, "", f"下载错误: {str(e)}"

    def download_image_task(self, task_data: Tuple[str, str, str]) -> Dict[str, Any]:
        """单个下载任务"""
        item_id, img_url, cover_dir = task_data
        
        # 检查图片是否已存在
        exists, _ = self.check_image_exists(item_id, cover_dir)
        if exists:
            return {"status": "existing", "item_id": item_id, "message": ""}
        
        # 检查URL是否有效
        if (
            pd.isna(img_url)
            or not img_url
            or str(img_url).strip() == ""
            or str(img_url).lower() == "nan"
        ):
            return {"status": "invalid_url", "item_id": item_id, "message": "URL无效"}
        
        # 下载图片
        save_path_base = os.path.join(cover_dir, item_id)
        success, final_path, message = self.download_single_image(img_url, save_path_base)
        
        if success:
            return {"status": "downloaded", "item_id": item_id, "message": message}
        else:
            return {"status": "failed", "item_id": item_id, "message": message}

    def download_dataset_images(
        self, item_pairs_file: str, cover_dir: str, dataset_name: str = ""
    ) -> Dict[str, int]:
        """多线程下载数据集的所有图片"""
        logger.info(f"开始下载 {dataset_name} 数据集图片 (使用{self.max_workers}个并发线程)")

        if not os.path.exists(item_pairs_file):
            logger.error(f"商品映射文件不存在: {item_pairs_file}")
            return {}

        os.makedirs(cover_dir, exist_ok=True)

        try:
            df = pd.read_csv(item_pairs_file)
        except Exception as e:
            logger.error(f"读取商品映射文件失败: {e}")
            return {}

        if "itemID" not in df.columns or "img_url" not in df.columns:
            logger.error("商品映射文件缺少必要列")
            return {}

        # 预处理：过滤已存在的图片和无效URL
        logger.info("预处理图片列表...")
        tasks = []
        for _, row in df.iterrows():
            item_id = str(row["itemID"])
            img_url = row["img_url"]
            
            # 预先检查是否已存在
            exists, _ = self.check_image_exists(item_id, cover_dir)
            if not exists and not (pd.isna(img_url) or not img_url or str(img_url).strip() == "" or str(img_url).lower() == "nan"):
                tasks.append((item_id, img_url, cover_dir))

        total_tasks = len(tasks)
        logger.info(f"需要下载的图片数量: {total_tasks}")
        
        if total_tasks == 0:
            logger.info("没有需要下载的图片")
            return {"existing": len(df), "downloaded": 0, "failed": 0, "invalid_url": 0}

        stats = {"existing": len(df) - total_tasks, "downloaded": 0, "failed": 0, "invalid_url": 0}
        
        # 使用多线程下载
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self.download_image_task, task): task for task in tasks}
            
            # 使用tqdm显示进度
            completed = 0
            with tqdm(total=total_tasks, desc=f"下载{dataset_name}图片", unit="张") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    stats[result["status"]] += 1
                    
                    if result["status"] == "failed":
                        logger.debug(f"下载失败 - ItemID: {result['item_id']}, 错误: {result['message']}")
                    
                    completed += 1
                    pbar.update(1)
                    
                    # 更新进度条描述，显示速度
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = completed / elapsed
                        pbar.set_postfix({
                            "成功": stats["downloaded"],
                            "失败": stats["failed"], 
                            "速度": f"{speed:.1f}张/秒"
                        })
                    
                    # 动态延时：优化速度
                    if result["status"] == "downloaded":
                        time.sleep(self.delay * 0.5)  # 成功后较短延时
                    elif result["status"] == "failed":
                        # 检查是否是404错误
                        if "404" in result.get("message", ""):
                            time.sleep(self.delay * 0.1)  # 404错误极短延时
                        else:
                            time.sleep(self.delay * 2)  # 其他失败稍长延时
                    else:
                        time.sleep(self.delay)  # 其他情况正常延时

        elapsed_time = time.time() - start_time
        avg_speed = total_tasks / elapsed_time if elapsed_time > 0 else 0

        # 打印详细统计信息
        print(f"\n{dataset_name} 图片下载完成:")
        print(f"  已存在: {stats['existing']}")
        print(f"  成功下载: {stats['downloaded']}")
        print(f"  下载失败: {stats['failed']}")
        print(f"  无效URL: {stats['invalid_url']}")
        print(f"  总耗时: {elapsed_time:.1f}秒")
        print(f"  平均速度: {avg_speed:.1f}张/秒")
        
        # 清理Session池
        with self.lock:
            for session in self.session_pool.values():
                session.close()
            self.session_pool.clear()

        return stats