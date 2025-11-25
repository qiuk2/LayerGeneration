#!/bin/bash

cd t2i-r1/src
RUN_NAME="t2i-r1"

export DEBUG_MODE="true"
export LOG_PATH="./outputs/debug.txt"
# export NCCL_DEBUG=INFO

QWEN_PATH="deepseek-ai/Janus-Pro-7B"
HF_DATASET="../../../data/geneval_and_t2i_data_final.json" 
OUTPUT_DIR="janus/outputs/${RUN_NAME}" 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="2,3" \
torchrun --nproc_per_node="2" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12345" \
open_r1/grpo.py --use_vllm False \
--deepspeed "../configs/zero3.json" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $QWEN_PATH \
--dataset_name $HF_DATASET \
--max_prompt_length 512 \
--max_completion_length 1024 \
--temperature 1.0 \
--num_generations 8 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--logging_steps 1 \
--bf16  \
--torch_dtype bfloat16 \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_steps 1600 \
--run_name $RUN_NAME \
--save_steps 400 \
--new_generations_image 1 \
--image_token_num_per_image 576 \
--cfg_weight 5 \
--reasoning_prompt_path ../../../data/prompt/reasoning_prompt.txt \
--reward_funcs orm \
--beta 0.01 \
--tf32 true \
--learning_rate 1e-6 \
--orm_ckpt_path ../../../reward_weight/ORM-T2I-R1
# --use_peft true \
# --lora_task_type CAUSAL_LM --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
# --use_rslora true \
# --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
