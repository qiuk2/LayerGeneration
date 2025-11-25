import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
parser.add_argument("--reasoning_prompt_path", type=str, default="../../../data/prompt/reasoning_prompt.txt")
parser.add_argument("--save_dir", type=str, default='', help="Path to the data directory")
parser.add_argument("--num_generation", type=int, default=4)

args = parser.parse_args()

# specify the path to the model
model_path = args.model_path
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

prompt_list = []
with open(args.data_path, 'r') as f:
    for line in f:
        prompt_list.append(line.strip())

with open(args.reasoning_prompt_path, 'r') as f:
    cot_prompt = f.read().strip()


def get_caption_height(text, font, img_width, draw):
    """Calculate the height needed for given text at specified width"""
    # Split text into words and handle line breaks
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Use textlength instead of textsize to get text width
        text_width = draw.textlength(test_line, font=font)
        
        if text_width < img_width - 20:  # 20 pixels margin
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Calculate required height based on number of lines
    try:
        font_size = font.size
    except:
        font_size = font.getsize('X')
        font_size = max(font_size)
    line_height = font_size + 4 # 4 pixel spacing between lines
    return len(lines) * line_height + 20  # 10 pixels margin at top and bottom

def create_grid_with_captions(visual_img, answer_list, save_dir, prompt_text, num_generation):
    """
    Create a grid of images with captions, all caption areas have the same height based on the longest caption
    
    Args:
        visual_img: List of numpy arrays containing images
        answer_list: List of caption texts for each image
        save_dir: Directory to save the output
        prompt_text: Prompt text used to generate the images
        num_generation: Number of images
    """
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    
    # Create a list to store images with captions
    captioned_images = []
    
    # Sample image to get width
    sample_img = Image.fromarray(visual_img[0])
    img_width, _ = sample_img.size
    
    # Set font
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # If arial font is not available, use default font
        font = ImageFont.load_default()
    try:
        font_size = font.size
    except:
        font_size = font.getsize('X')
        font_size = max(font_size)
    
    # Create temporary image for text calculations
    temp_img = Image.new('RGB', (img_width, 200))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Calculate maximum caption height needed
    max_caption_height = 0
    for i in range(min(len(answer_list), num_generation)):
        caption = answer_list[i]
        caption_height = get_caption_height(caption, font, img_width, temp_draw)
        max_caption_height = max(max_caption_height, caption_height)
    
    # Ensure there's a minimum height
    max_caption_height = max(max_caption_height, 30)
    print(f"Maximum caption height: {max_caption_height} pixels")
    
    # Process each image using the calculated maximum height
    for i in range(num_generation):
        # Get original image
        img = Image.fromarray(visual_img[i])
        img_width, img_height = img.size
        
        # Get caption text
        caption = answer_list[i] if i < len(answer_list) else ""
        
        # Create new image with fixed caption space
        captioned_img = Image.new('RGB', (img_width, img_height + max_caption_height), color='white')
        
        # Paste original image
        captioned_img.paste(img, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(captioned_img)
        
        # Split text into lines to fit image width
        words = caption.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            text_width = draw.textlength(test_line, font=font)
            
            if text_width < img_width - 20:  # 20 pixels margin
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw text in caption area
        line_height = font_size + 4
        total_text_height = len(lines) * line_height
        # y_text = img_height + (max_caption_height - total_text_height) // 2  # Vertical center
        y_text = img_height + 10  # Vertical center
        
        for line in lines:
            text_width = draw.textlength(line, font=font)
            x_position = (img_width - text_width) // 2  # Horizontal center
            draw.text((x_position, y_text), line, fill="black", font=font)
            y_text += line_height
        
        # Convert to tensor for grid creation
        captioned_tensor = torch.from_numpy(np.array(captioned_img)).permute(2, 0, 1)
        captioned_images.append(captioned_tensor)
    
    # Create square grid
    nrow = int(np.ceil(np.sqrt(num_generation)))
    grid = torchvision.utils.make_grid(captioned_images, nrow=nrow)
    grid = grid.permute(1, 2, 0).numpy()  # Convert back to (H, W, C) format
    grid = grid.astype(np.uint8)  # Ensure correct data type
    
    # Save grid
    os.makedirs(save_dir, exist_ok=True)
    grid_path = os.path.join(save_dir, prompt_text.replace(' ', '_') + ".jpg")
    print(grid_path)
    PIL.Image.fromarray(grid).save(grid_path)
    
    return grid_path

def to_pil_list(imgs):
    pil_imgs = []
    for im in imgs:
        # torch -> numpy
        if torch.is_tensor(im):
            im = im.detach().cpu()
            # [3,H,W] -> [H,W,3]
            if im.ndim == 3 and im.shape[0] in (1,3):
                im = im.permute(1,2,0)
            im = im.numpy()

        # numpy -> uint8 HWC
        if isinstance(im, np.ndarray):
            # 如果是 float，先 clip 再转 uint8
            if im.dtype != np.uint8:
                im = np.clip(im, 0, 255).astype(np.uint8)
            # 保证 HWC
            if im.ndim == 3 and im.shape[0] in (1,3) and im.shape[-1] not in (1,3):
                im = np.transpose(im, (1,2,0))
            im = Image.fromarray(im)

        # 兜底：如果已经是 PIL 直接用
        if not isinstance(im, Image.Image):
            raise TypeError(f"Unsupported image type: {type(im)}")

        pil_imgs.append(im)
    return pil_imgs



@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_text: str,
    temperature: float = 1,
    num_generation: int = 9,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):  
    device = 'cuda'

    def sample_image_tokens_cfg(
        cond_inputs_embeds: torch.Tensor,      # [G, T, D]
        cond_attention_mask: torch.Tensor,     # [G, T]
    ) -> torch.Tensor:
        """Return generated image token ids of shape [G, image_token_num_per_image]."""
        G = cond_inputs_embeds.size(0)

        # uncond embeds = pad out text region, keep BOS and last (image start)
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        pad_token = torch.tensor([[vl_chat_processor.pad_id]], device=device)
        pad_embed = mmgpt.language_model.get_input_embeddings()(pad_token)  # [1,1,D]
        if uncond_inputs_embeds.size(1) > 2:
            uncond_inputs_embeds[:, 1:-1] = pad_embed

        # interleave cond/uncond
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)  # [2G, T, D]
        inputs_embeds_img[1::2] = uncond_inputs_embeds

        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

        generated_tokens = torch.zeros((G, image_token_num_per_image), dtype=torch.int64, device=device)

        cur_inputs_embeds_img = inputs_embeds_img
        cur_attention_mask_img = attention_mask_img
        past = None

        for k in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(
                inputs_embeds=cur_inputs_embeds_img,
                attention_mask=cur_attention_mask_img,
                use_cache=True,
                past_key_values=past,
            )
            past = outputs.past_key_values

            hidden_states = outputs.last_hidden_state  # [2G, T, D]
            logits = mmgpt.gen_head(hidden_states[:, -1, :])  # [2G, V]

            logit_cond = logits[0::2]
            logit_uncond = logits[1::2]
            guided = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(guided / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [G,1]
            generated_tokens[:, k] = next_token.squeeze(-1)

            # expand to cond/uncond order
            next_token_2 = torch.cat(
                [next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1
            ).view(-1)  # [2G]

            img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2)  # [2G, D]
            cur_inputs_embeds_img = img_embeds.unsqueeze(1)          # [2G, 1, D]

            cur_attention_mask_img = torch.cat(
                [cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.size(0), 1))],
                dim=1,
            )

        return generated_tokens

    prompt_inputs = vl_chat_processor.tokenizer(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
    )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    prompt_ids = prompt_ids.repeat_interleave(num_generation, dim=0).to(device)
    prompt_mask = prompt_mask.repeat_interleave(num_generation, dim=0).to(device)
    input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)


    # TODO: if num_generations is too large, we need to split it into multiple batches
    if num_generation > 20:
        total_generations = []
        for i in range(prompt_ids.shape[0] // num_generation):
            current_input_embeds = input_embeds[i*num_generation: (i+1)*num_generation]
            current_attn_mask = prompt_mask[i*num_generation: (i+1)*num_generation]
            prompt_completion_ids = mmgpt.language_model.generate(
                inputs_embeds=current_input_embeds,
                attention_mask=current_attn_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )
            total_generations.append(prompt_completion_ids)
        prompt_completion_ids = torch.cat(total_generations, dim=0)
    else: # if num_generations == 1, we directly generate all for the batch data
        prompt_completion_ids = mmgpt.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=prompt_mask,
            pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
            bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
            eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            use_cache=True,
        )

    prompt_length = prompt_ids.size(1)
    completion_ids = prompt_completion_ids

    # =========================================================
    # (B) parse layer prompts
    # =========================================================
    image_gen_prompt_list = []
    
    global_prompt_str = vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
    for i in range(completion_ids.shape[0]):
        answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
        lines = [l.strip() for l in answer.splitlines() if l.strip()]

        if len(lines) >= 2:
            layer_lines = lines[-2:]
            reasoning_lines = lines[:-2]
        else:
            layer_lines = lines
            reasoning_lines = []
        reasoning_text = " ".join(reasoning_lines).strip()

        for j, layer in enumerate(layer_lines):
            if reasoning_text:
                image_gen_prompt = f"{reasoning_text} {layer}"
            else:
                image_gen_prompt = f"{layer}"

            conversation = [
                {"role": "User", "content": image_gen_prompt},
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            image_gen_prompt_list.append(sft_format)
            print(f"layer {j}: {sft_format}")

    # =========================================================
    # (C) layer prompts -> layer image tokens（每层并行的 teacher-forcing 结构）
    # =========================================================
    prompt_inputs = vl_chat_processor.tokenizer(
        text=image_gen_prompt_list,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    )
    layer_prompt_ids = prompt_inputs["input_ids"].to(device)
    layer_attn_mask = prompt_inputs["attention_mask"].to(device)

    image_start_token_id = vl_chat_processor.tokenizer.encode(
        vl_chat_processor.image_start_tag
    )[1]
    layer_prompt_ids = torch.cat(
        [layer_prompt_ids, layer_prompt_ids.new_full((layer_prompt_ids.size(0), 1), image_start_token_id)],
        dim=1,
    )
    layer_attn_mask = torch.cat(
        [layer_attn_mask, layer_attn_mask.new_ones((layer_attn_mask.size(0), 1))],
        dim=1,
    )

    layer_inputs_embeds = mmgpt.language_model.get_input_embeddings()(layer_prompt_ids)

    total_generated_tokens_img = []
    num_layers = layer_inputs_embeds.size(0) // num_generation

    for j in range(num_layers):
        cond_embeds = layer_inputs_embeds[j * num_generation : (j + 1) * num_generation]
        cond_mask   = layer_attn_mask[j * num_generation : (j + 1) * num_generation]
        gen_tokens  = sample_image_tokens_cfg(cond_embeds, cond_mask)  # [G, 576]
        total_generated_tokens_img.append(gen_tokens)

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)
    # [G*num_layers, 576]

    # decode to RGB images for each layer
    H = img_size // patch_size
    layer_dec = mmgpt.gen_vision_model.decode_code(
        total_generated_tokens_img.to(dtype=torch.int),
        shape=[num_generation * num_layers, 8, H, H],
    )
    layer_dec = layer_dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    layer_dec = np.clip((layer_dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    layer_images = layer_dec.reshape(num_generation * num_layers, img_size, img_size, 3)

    # =========================================================
    # (D) compose layer images -> understanding tokens
    # =========================================================
    # 这里我给一个最简 compose：按层平均。
    # 你后面可以替换成 mask/alpha 的正确 compositing。
    composed_images = layer_images.astype(np.uint8)  # [G,H,W,3]
    create_grid_with_captions(
        composed_images,
        image_gen_prompt_list,
        args.save_dir,
        prompt_text,
        num_generation * num_layers,
    )
    print("shape", composed_images.shape)

    print("prompt", prompt_text)

    conversation = [
            {
                "role": "User",
                "content": f"{prompt_text}. Layer 1:<image_placeholder>, Layer 2:<image_placeholder>",
                "images": to_pil_list(composed_images)
            },
            {"role": "Assistant", "content": ""},
        ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
    print("sft format", sft_format)

    final_prompt_inputs = vl_chat_processor.tokenizer(
        text=sft_format,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    ) 

    final_prompt_ids, final_attention_mask = final_prompt_inputs["input_ids"], final_prompt_inputs["attention_mask"]
    final_prompt_ids = final_prompt_ids.to(device)
    final_attention_mask = final_attention_mask.to(device)
    print("final prom", final_prompt_ids.shape)

    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    final_prompt_ids = torch.cat([final_prompt_ids, final_prompt_ids.new_full((final_prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    final_attention_mask = torch.cat([final_attention_mask, final_attention_mask.new_ones((final_attention_mask.size(0), 1))], dim=1)
    
    final_inputs_embeds = mmgpt.language_model.get_input_embeddings()(final_prompt_ids)
    final_tokens = sample_image_tokens_cfg(final_inputs_embeds, final_attention_mask)  # [G,576]

    # decode final
    final_dec = mmgpt.gen_vision_model.decode_code(
        final_tokens.to(dtype=torch.int),
        shape=[num_generation, 8, H, H],
    )
    final_dec = final_dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    final_dec = np.clip((final_dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # 可选存图/网格
    if args.save_dir is not None:
        # 你如果有自己的可视化函数可替换这里
        create_grid_with_captions(
            final_dec,
            [global_prompt_str] * num_generation,
            args.save_dir,
            prompt_text + " [FINAL]",
            num_generation,
        )

    return {
        "text_completions": completion_ids,
        "layer_prompts": image_gen_prompt_list,
        "layer_tokens": total_generated_tokens_img,
        "layer_images": layer_images,              # [G, num_layers, H, W, 3]
        "composed_images": composed_images,        # [G, H, W, 3]
        "final_tokens": final_tokens,              # [G, 576]
        "final_images": final_dec,                 # [G, H, W, 3]
    }

random.shuffle(prompt_list)
for prompt in prompt_list:
    prompt_text = copy.deepcopy(prompt)
    conversation = [
        {
            "role": "User",
            "content": cot_prompt.format(prompt),
        },
        {"role": "Assistant", "content": ""},
    ]

    system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=system_prompt,
    )
    prompt = sft_format

    generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        prompt_text,
        num_generation=args.num_generation,
        conversation=conversation
    )