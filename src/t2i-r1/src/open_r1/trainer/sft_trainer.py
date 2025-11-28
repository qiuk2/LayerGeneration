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

'''
Two Forward Passes
'''

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, List, Dict
from PIL import Image

import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from janus.models import MultiModalityCausalLM, VLChatProcessor


class JanusSFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        args = None,
    ):
        # Models
        # Trained model
        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            # torch_dtype = model_init_kwargs.get("torch_dtype")
            # if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            #     pass  # torch_dtype is already a torch.dtype or "auto" or None
            # elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            #     torch_dtype = getattr(torch, torch_dtype)
            #     model_init_kwargs["torch_dtype"] = torch_dtype
            # else:
            #     raise ValueError(
            #         "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
            #         f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            #     )
            # # Disable caching if gradient checkpointing is enabled (not supported)
            # model_init_kwargs["use_cache"] = (
            #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            # )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, use_safetensors=True
            )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        model.language_model.config._attn_implementation == "flash_attention_2"
        # freeze all vision encoders
        for name, param in model.named_parameters():
            if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"): # choose whatever you like here
                param.requires_grad = False
        # try gradient checkpointing
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable()
        # remove unnecessary parameters
        # del model.vision_model
        # del model.aligner
        print("current lora config", peft_config)
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Processing class
        if processing_class is None:
            processing_class: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)
        self.processing_class = processing_class

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        # self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device

        # -------- A. 构造 prompt text + cot text --------
        def sample_one_prompt(ex):
            # 在 [0, 4] 之间随机一个整数
            r = torch.randint(0, 4, ()).item()  # 标量 int

            if r == 0:
                # 1: layer_prompt
                return ex["layer_prompt"]
            elif r == 1:
                # 2: generation_prompt_1
                return ex["generation_prompt_1"]
            elif r == 2:
                # 3: generation_prompt_2
                return ex["generation_prompt_2"]
            else:
                # 4: combination_prompt
                return ex["combination_prompt"]

        
        prompts = [sample_one_prompt(ex) for ex in inputs]

        prompts_text = [
            self.processing_class.apply_sft_template_for_multi_turn_prompts(
            conversations=prompt,
            sft_format=self.processing_class.sft_format,
            system_prompt="",
        ) for prompt in prompts]
        prompt_inputs= self.processing_class.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left", # left padding, same in the official repo
            add_special_tokens=True,
        ) # {'input_ids', 'attention_mask'}
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        labels = prompt_ids.clone()
        pad_token_id = self.processing_class.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[prompt_ids == pad_token_id] = -100

        # 也可以用 attention_mask 再保险一下：
        labels[prompt_mask == 0] = -100

        # -------- D. 前向计算 loss --------
        # 这里直接用底层 language_model（一个标准 CausalLM）
        outputs = model.language_model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            labels=labels,
        )
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss
