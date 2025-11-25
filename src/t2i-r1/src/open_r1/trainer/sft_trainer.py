import torch
import torch.nn.functional as F
from transformers import Trainer

class JanusSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        processing_class=None,            # VLChatProcessor
        image_start_token_id=None,        # processor.image_start_tag 对应 id
        max_prompt_length=None,
        max_textcot_length=None,
        image_loss_weight=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert processing_class is not None
        self.processing_class = processing_class
        self.image_start_token_id = image_start_token_id
        self.max_prompt_length = max_prompt_length
        self.max_textcot_length = max_textcot_length
        self.image_loss_weight = image_loss_weight

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device

        # -------- A. 构造 prompt text + cot text --------
        prompts = [ex["prompt"] for ex in inputs]
        cots = [ex.get("cot", "") for ex in inputs]

        prompts_text = [
            self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=p,
                sft_format=self.processing_class.sft_format,
                system_prompt="You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.",
            )
            for p in prompts
        ]

        # full text = prompt + cot
        full_text = [pt + cot for pt, cot in zip(prompts_text, cots)]

        # -------- B. tokenization（批量）--------
        tok_full = self.processing_class.tokenizer(
            text=full_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        ).to(device)

        input_ids = tok_full["input_ids"]
        attn_mask = tok_full["attention_mask"]

        # （可选）截断 prompt 长度
        if self.max_prompt_length is not None:
            input_ids = input_ids[:, -self.max_prompt_length:]
            attn_mask = attn_mask[:, -self.max_prompt_length:]

        # -------- C. 只监督 cot：mask prompt 部分 labels --------
        # 关键：每个样本 prompt_len 可能不同，所以逐样本算长度
        prompt_lens = []
        for pt in prompts_text:
            ids = self.processing_class.tokenizer(
                pt, add_special_tokens=True, return_tensors=None
            )["input_ids"]
            prompt_lens.append(len(ids))

        labels = input_ids.clone()
        # left padding 情况下，prompt 在右侧对齐，所以要从右往左 mask
        for i, plen in enumerate(prompt_lens):
            # 取该样本有效长度（除去 pad）
            valid_len = attn_mask[i].sum().item()
            # prompt 在有效区间开头，cot 在后面
            start = input_ids.size(1) - valid_len
            labels[i, start : start + plen] = -100

        # （可选）限制 cot 最大长度（从尾部截）
        if self.max_textcot_length is not None:
            # 只简单裁 full 序列尾部
            input_ids = input_ids[:, -(self.max_textcot_length + max(prompt_lens)):]
            attn_mask = attn_mask[:, -(self.max_textcot_length + max(prompt_lens)):]
            labels = labels[:, -(self.max_textcot_length + max(prompt_lens)):]

        # -------- D. 文本 CE loss --------
        out_text = model.language_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        text_logits = out_text.logits  # (B, L, V)

        text_loss = F.cross_entropy(
            text_logits[:, :-1].contiguous().view(-1, text_logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

        # -------- E. 图像 token CE loss（可选）--------
        img_loss = torch.tensor(0.0, device=device)
        if "image_tokens" in inputs[0]:
            # (B, T)
            img_ids = torch.stack([
                torch.as_tensor(ex["image_tokens"], dtype=torch.long)
                for ex in inputs
            ]).to(device)

            # teacher forcing: 输入 = <image_start> + img_ids[:-1]
            image_start = torch.full(
                (img_ids.size(0), 1),
                self.image_start_token_id,
                dtype=torch.long,
                device=device
            )
            img_in = torch.cat([image_start, img_ids[:, :-1]], dim=1)  # (B, T)

            # embeddings
            text_embeds = model.language_model.get_input_embeddings()(input_ids)
            img_embeds = model.prepare_gen_img_embeds(img_in)
            all_embeds = torch.cat([text_embeds, img_embeds], dim=1)

            img_attn = torch.ones_like(img_in, device=device)
            all_mask = torch.cat([attn_mask, img_attn], dim=1)

            # forward -> hidden states
            hs = model.language_model(
                inputs_embeds=all_embeds,
                attention_mask=all_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]

            # 最后 T 个位置对应图像 token 预测
            image_logits = model.gen_head(hs[:, -img_in.size(1):, :])  # (B, T, V_img)

            img_loss = F.cross_entropy(
                image_logits.contiguous().view(-1, image_logits.size(-1)),
                img_ids.contiguous().view(-1),
            )

        loss = text_loss + self.image_loss_weight * img_loss
        return (loss, out_text) if return_outputs else loss
