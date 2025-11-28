#!/bin/bash

cd t2i-r1/src
RUN_NAME="t2i-r1"

export DEBUG_MODE="true"
export LOG_PATH="./outputs/debug.txt"
# export NCCL_DEBUG=INFO

QWEN_PATH="deepseek-ai/Janus-Pro-7B"
HF_DATASET="/scr/kaiq/meta.csv" 
OUTPUT_DIR="janus/outputs/${RUN_NAME}" 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="4,5" \
torchrun --nproc_per_node="2" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12344" \
open_r1/sft.py --use_vllm False \
--deepspeed "../configs/zero3.json" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $QWEN_PATH \
--dataset_name $HF_DATASET \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--logging_steps 1 \
--bf16  \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_steps 20000 \
--run_name $RUN_NAME \
--save_steps 10000 \
--image_token_num_per_image 576 \
--layer_composition_prompt_path ../../../data/prompt/layer_prompt.txt \
--combination_prompt_path ../../../data/prompt/combination_prompt.txt \
--beta 0.01 \
--tf32 true \
--learning_rate 1e-6 \
--remove_unused_columns false
# --use_peft true \
# --lora_task_type CAUSAL_LM --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
# --use_rslora true \
# --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
