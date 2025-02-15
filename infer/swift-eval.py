import os
import json
import subprocess
import sys
from datetime import datetime
import logging
from typing import List, Dict
import re
import time
import threading
from threading import Event, Lock

import torch._dynamo
torch._dynamo.config.suppress_errors = True

class GPUMonitorThread(threading.Thread):
    def __init__(
        self, 
        manager,
        check_interval: int = 60
    ):
        super().__init__()
        self.manager = manager
        self.check_interval = check_interval
        self.stop_event = Event()
        self.daemon = True
        
    def run(self):
        while not self.stop_event.is_set():
            memory_usage = self.manager.get_gpu_memory_usage()
            self.manager.logger.info(f"Current GPU memory usage: {memory_usage:.2%}")
            
            # 获取锁来修改 batch_size
            with self.manager.batch_size_lock:
                # 只有在没有发生 OOM 的情况下才考虑增加 batch_size
                if not self.manager.recent_oom and \
                   memory_usage < self.manager.mem_threshold and \
                   self.manager.batch_size < self.manager.initial_batch_size:
                    self.manager.batch_size = self.manager.initial_batch_size
                    self.manager.logger.info(f"GPU memory usage is low, restored batch size to {self.manager.batch_size}")
            
            self.stop_event.wait(self.check_interval)
    
    def stop(self):
        self.stop_event.set()

class SwiftInferenceManager:
    def __init__(
        self,
        model_path: str,
        input_data_paths: List[str],
        base_output_dir: str = "swift_eval",
        chunk_size: int = 100,
        initial_batch_size: int = 8,
        max_new_tokens: int = 2048,
        gpu_id: int = 0,
        mem_threshold: float = 0.3,
        monitor_interval: int = 60,
        oom_cooldown: int = 300  # OOM 冷却时间（秒）
    ):
        self.model_path = model_path
        self.input_data_paths = input_data_paths
        self.base_output_dir = base_output_dir
        self.chunk_size = chunk_size
        self.initial_batch_size = initial_batch_size
        self.batch_size = initial_batch_size
        self.max_new_tokens = max_new_tokens
        self.gpu_id = gpu_id
        self.mem_threshold = mem_threshold
        self.monitor_interval = monitor_interval
        self.oom_cooldown = oom_cooldown
        
        # 添加锁和 OOM 状态追踪
        self.batch_size_lock = Lock()
        self.recent_oom = False
        self.last_oom_time = 0
        
        self.setup_logging()
        os.makedirs(base_output_dir, exist_ok=True)
        # self.gpu_monitor = GPUMonitorThread(self, monitor_interval)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('inference.log')
            ]
        )
        self.logger = logging

    def get_gpu_memory_usage(self) -> float:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader', '-i', str(self.gpu_id)],
                capture_output=True,
                text=True,
                check=True
            )
            memory_used, memory_total = map(int, result.stdout.strip().split(','))
            return memory_used / memory_total
        except Exception as e:
            self.logger.error(f"Error getting GPU memory usage: {str(e)}")
            return 1.0

    def load_data(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            return []

    def save_chunk(self, data: List[Dict], chunk_dir: str, chunk_id: int):
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_id}.jsonl")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return chunk_path

    def run_swift_inference(self, input_path: str, output_path: str, log_file: str) -> bool:
        command = [
            "swift", "infer",
            "--model", self.model_path,
            "--load_data_args", "true",
            "--stream", "false",
            "--max_new_tokens", str(self.max_new_tokens),
            "--max_batch_size", str(self.batch_size),
            "--val_dataset", input_path,
            "--result_path", output_path,
            "--model_type", "deepseek_vl"
        ]
        
        try:
            with open(log_file, 'a') as f:
                subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
                    }
                )
            
            # # 推理成功，检查是否可以清除 OOM 状态
            # current_time = time.time()
            # if self.recent_oom and (current_time - self.last_oom_time) > self.oom_cooldown:
            #     with self.batch_size_lock:
            #         self.recent_oom = False
            #         self.logger.info("OOM cooldown period ended")
            
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
                with self.batch_size_lock:
                    self.batch_size = max(1, self.batch_size // 2)
                    self.recent_oom = True
                    self.last_oom_time = time.time()
                    self.logger.warning(f"OOM detected. Reduced batch size to {self.batch_size}")
                return False
            self.logger.error(f"Inference failed: {error_msg}")
            return False

    def merge_results(self, task_dir: str, final_output_path: str):
        merged_results = []
        for filename in sorted(os.listdir(task_dir)):
            if filename.startswith("chunk_") and filename.endswith("_results.jsonl"):
                with open(os.path.join(task_dir, filename), 'r', encoding='utf-8') as f:
                    # 逐行读取JSONL文件
                    for line in f:
                        if line.strip():  # 跳过空行
                            chunk_result = json.loads(line)
                            merged_results.append(chunk_result)
        
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)

    def process_task(self, input_path: str):
        task_name = os.path.basename(input_path).split('.')[0]
        task_dir = os.path.join(self.base_output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        data = self.load_data(input_path)
        if not data:
            return
        
        progress_file = os.path.join(task_dir, "progress.jsonl")
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            start_chunk = progress['last_completed_chunk'] + 1
        else:
            progress = {'last_completed_chunk': -1}
            start_chunk = 0
        
        chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        
        for chunk_id in range(start_chunk, len(chunks)):
            chunk_data = chunks[chunk_id]
            chunk_path = self.save_chunk(chunk_data, task_dir, chunk_id)
            chunk_output = os.path.join(task_dir, f"chunk_{chunk_id}_results.jsonl")
            log_file = os.path.join(task_dir, f"chunk_{chunk_id}.log")
            
            retry_count = 0
            while not self.run_swift_inference(chunk_path, chunk_output, log_file):
                retry_count += 1
                if self.batch_size < 1:
                    self.logger.error("Batch size reduced to 0. Cannot continue.")
                    return
                    
                # 添加重试等待时间，随重试次数增加而增加
                wait_time = min(30, retry_count * 5)  # 最多等待30秒
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            
            progress['last_completed_chunk'] = chunk_id
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        final_output = os.path.join(self.base_output_dir, f"{task_name}_final_results.jsonl")
        self.merge_results(task_dir, final_output)
        self.logger.info(f"Task {task_name} completed. Results saved to {final_output}")

    def run(self):
        # 启动GPU监控线程
        # self.gpu_monitor.start()
        
        try:
            for input_path in self.input_data_paths:
                self.logger.info(f"Processing task: {input_path}")
                self.process_task(input_path)
        finally:
            # 确保在处理完成或发生异常时停止监控线程
            # self.gpu_monitor.stop()
            # self.gpu_monitor.join()
            pass

def main():
    ##要改的
    model_path = ""
    base_output_dir=""
    #############
    input_data_paths = [
        "{your_test_set}/SARChat_classify_count_test.json",
        "{your_test_set}/SARChat_classify_grounding_multi_test.json",
        "{your_test_set}/SARChat_classify_grounding_single_test.json",
        "{your_test_set}/SARChat_classify_identify_multi_test.json",
        "{your_test_set}/SARChat_classify_identify_single_test.json",
        "{your_test_set}/SARChat_classify_refer_multi_test.json",
        "{your_test_set}/SARChat_classify_refer_single_test.json",
        "{your_test_set}/SARChat_classify_test.json",
        "{your_test_set}/SARChat_describe_test.json",
    ]
    
    manager = SwiftInferenceManager(
        model_path=model_path,
        input_data_paths=input_data_paths,
        base_output_dir=base_output_dir,
        chunk_size=10000,
        initial_batch_size=8,
        max_new_tokens=2048,
        gpu_id=0,
        mem_threshold=0.3,
        monitor_interval=60,
        oom_cooldown=600  # 10分钟的 OOM 冷却期
    )
    manager.run()

if __name__ == "__main__":
    main()