# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
from transformers import HfArgumentParser, TrainingArguments, get_scheduler
from janus.models import MultiModalityCausalLM, VLChatProcessor
from open_r1.trainer import JanusSFTTrainer

@dataclass
class TrainingArguments(TrainingArguments):
    layer_composition_prompt_path: Optional[str] = field(
        default='',
    )
    combination_prompt_path: Optional[str] = field(
        default='',
    )
    attn_implementation: str = "flash_attention_2"
    dataset_name: Optional[str] = field(
        default='',
    )
    model_name_or_path: str = "deepseek-ai/Janus-Pro-7B"
    use_vllm: bool = False
    image_token_num_per_image: int = 576
    beta: float = 0.01
    max_prompt_length: int = 1024
    


def main(args):
    # Load the dataset
    if args.dataset_name.endswith('.csv'):
        suffix = 'csv'
    elif args.dataset_name.endswith('.json'):
        suffix = 'json'
    elif args.dataset_name.endswith('.parquet'):
        suffix = 'parquet'
    dataset = load_dataset(suffix, data_files=args.dataset_name)
    print('Dataset length: ', len(dataset['train']))

    # load cot prompt
    if args.layer_composition_prompt_path:
        with open(args.layer_composition_prompt_path, 'r') as f:
            layer_prompt = f.read()
            args.layer_prompt = layer_prompt

    if args.combination_prompt_path:
        with open(args.combination_prompt_path, 'r') as f:
            combination_prompt = f.read()
            args.combination_prompt = combination_prompt
    
            
    # Format into conversation
    def make_conversation(example):
        # make detection prompt
        if 'nouns' in example and example['nouns'] is not None:
            det_text_prompt, det_token_spans = make_detection_prompt(example['nouns'])
        else:
            det_text_prompt = ''
            det_token_spans = []
        det_prompt_dict = {
            'text_prompt': det_text_prompt,
            'token_spans': det_token_spans,
        }
        # make vqa prompt
        if 'attr_nouns' in example and example['attr_nouns'] is not None:
            questions = [f"{attr_noun}?" for attr_noun in example['attr_nouns']]
            vqa_prompt = {'questions': questions}
        else:
            vqa_prompt = {'questions': []}  # Changed from None to empty list

        return {
            "prompt": [
                {"role": "User", "content": cot_prompt.format(example["prompt"])},
                {"role": "Assistant", "content": ""},
            ],
            'raw_prompt': example["prompt"],
            'det_prompt': det_prompt_dict,
            'task_type': example['task_type'],
        }

    def make_conversation_image(example):
        return {
            "layer_prompt": [
                {
                    "role": "User",
                    "content": layer_prompt.format(example['global_prompt']),
                },
                {"role": "Assistant", "content": f"{example['background_prompt']}\n{example['foreground_prompt']}"},
            ],
            "generation_prompt_1": [
                {
                    "role": "User",
                    "content": example['background_prompt'],
                },
                {"role": "Assistant", "content": "", "images": [example['image2_path']],},
            ],
            "generation_prompt_2": [
                {
                    "role": "User",
                    "content": example['foreground_prompt'],
                },
                {"role": "Assistant", "content": "", "images": [example['image1_path']],},
            ],
            "combination_prompt": [
                {
                    "role": "User",
                    "content": combination_prompt.format(example['global_prompt']),
                    "images": [example['image2_path'], example['image1_path']],
                },
                {
                    "role": "Assistant",
                    "content": "",
                    "images": [example['final_image_path']],
                },
            ]
        }



    # if "image" in dataset[args.dataset_train_split].features or 'image_path' in dataset[args.dataset_train_split].features:
    print("***************has image in dataset***************")
    dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    # else:
    #     print("***************no image in dataset***************")
    #     dataset = dataset.map(
    #         make_conversation,
    #         num_proc=1,
    #         # remove_columns=['spatial_info', 'numeracy_info', 'attr_nouns', 'nouns']
    #     )
    #     # dataset = dataset.remove_columns("messages")

    
    trainer_cls = JanusSFTTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=args.model_name_or_path,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        attn_implementation=args.attn_implementation,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(dataset_name=args.dataset_name)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    args, = parser.parse_args_into_dataclasses()
    main(args)
