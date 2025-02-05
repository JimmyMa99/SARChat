#!/bin/bash

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/[SARChat-TOTAL]Yi_vl_6b_lora_sft_${TIMESTAMP}.log"

# 设置环境变量
export NPROC_PER_NODE=2
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

nohup swift sft \
    --model {set/your/model/path}/01ai/Yi-VL-6B/01ai/Yi-VL-6B \
    --train_type lora \
    --dataset '{set/your/dataset/path}sarchat-exp/data/SARChat_total_task_train_modified.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 10000 \
    --save_total_limit 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ./swift_output/SARChat-Yi-VL-6B-Lora \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 16 \
    --model_author JimmyMa99 \
    --model_name SARChat-Yi-VL-6B-Lora \
    --resume_from_checkpoint {set/your/dataset/path}sarchat-exp/swift_output/SARChat-Yi-VL-6B-Lora/v3-20250204-030621/checkpoint-10000 \
    > "$LOG_FILE" 2>&1 &

# 打印进程ID和日志文件位置
echo "Training started with PID $!"
echo "Log file: $LOG_FILE"

# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"