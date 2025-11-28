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
import re
from PIL import Image

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


os.makedirs(args.save_dir, exist_ok=True)



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

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_text: str,
    temperature: float = 1.0,
    num_generation: int = 9,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):
    device = "cuda"

    # ===== 1. 文本 prompt -> ids =====
    prompt_inputs = vl_chat_processor.tokenizer(
        text=[prompt],
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    )
    prompt_ids = prompt_inputs["input_ids"].to(device)          # [1, T]
    attention_mask = prompt_inputs["attention_mask"].to(device) # [1, T]

    # 扩展成 num_generation 个相同 prompt
    prompt_ids = prompt_ids.repeat(num_generation, 1)           # [G, T]
    attention_mask = attention_mask.repeat(num_generation, 1)   # [G, T]

    # ===== 2. 末尾加 image_start token =====
    image_start_token_id = vl_chat_processor.tokenizer.encode(
        vl_chat_processor.image_start_tag
    )[1]
    image_start_tokens = prompt_ids.new_full(
        (prompt_ids.size(0), 1), image_start_token_id
    )                                                           # [G, 1]
    prompt_ids = torch.cat([prompt_ids, image_start_tokens], dim=1)  # [G, T+1]

    attention_mask = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))],
        dim=1,
    )  # [G, T+1]

    # 文本+image_start 的 embedding
    cond_inputs_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)  # [G, T+1, D]

    # ===== 3. 构造 uncond 版本（CFG） =====
    pad_token_id = vl_chat_processor.pad_id
    pad_token = torch.full((1, 1), pad_token_id, dtype=torch.long, device=device)
    pad_embed = mmgpt.language_model.get_input_embeddings()(pad_token)  # [1,1,D]

    uncond_inputs_embeds = cond_inputs_embeds.clone()  # [G, T+1, D]
    if uncond_inputs_embeds.size(1) > 2:
        # 保留 BOS 和最后一个 image_start，中间全部用 pad_embed
        uncond_inputs_embeds[:, 1:-1] = pad_embed

    # 交错堆叠 cond / uncond -> [2G, T+1, D]
    G, T_plus_1, D = cond_inputs_embeds.shape
    inputs_embeds_img = torch.empty(
        (2 * G, T_plus_1, D),
        dtype=cond_inputs_embeds.dtype,
        device=device,
    )
    inputs_embeds_img[0::2] = cond_inputs_embeds
    inputs_embeds_img[1::2] = uncond_inputs_embeds

    attention_mask_img = torch.empty(
        (2 * G, attention_mask.size(1)),
        dtype=attention_mask.dtype,
        device=device,
    )
    attention_mask_img[0::2] = attention_mask
    attention_mask_img[1::2] = 1  # uncond 全开

    # ===== 4. 自回归生成 image tokens（batch 并行） =====
    generated_tokens = torch.zeros(
        (G, image_token_num_per_image), dtype=torch.long, device=device
    )

    cur_inputs_embeds = inputs_embeds_img          # [2G, T+1, D] for k=0
    cur_attention_mask = attention_mask_img        # [2G, T+1]
    past = None

    for k in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=cur_inputs_embeds,
            attention_mask=cur_attention_mask,
            use_cache=True,
            past_key_values=past,
        )
        past = outputs.past_key_values

        hidden_states = outputs.last_hidden_state  # [2G, seq, D]
        logits = mmgpt.gen_head(hidden_states[:, -1, :])  # [2G, V_img]

        logit_cond = logits[0::2]    # [G, V_img]
        logit_uncond = logits[1::2]  # [G, V_img]

        guided = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(guided / max(temperature, 1e-6), dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)  # [G, 1]
        generated_tokens[:, k] = next_token.squeeze(-1)       # 写入第 k 个位置

        # 下一个时间步的输入：把 cond/uncond 都喂同一个 token
        next_token_2 = next_token.expand(-1, 2).reshape(-1)   # [2G]
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2)  # [2G, D]

        cur_inputs_embeds = img_embeds.unsqueeze(1)           # [2G, 1, D]
        cur_attention_mask = torch.cat(
            [
                cur_attention_mask,
                cur_attention_mask.new_ones(
                    (cur_attention_mask.size(0), 1)
                ),
            ],
            dim=1,
        )  # 每一步把 mask 长度 +1

    # ===== 5. decode 所有 G 张图像 =====
    H = img_size // patch_size
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[G, 8, H, H],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = dec  # [G, img_size, img_size, 3]

    # 下面这行你原来用的是 image_gen_prompt_list 和 args.save_dir，
    # 这里假设你在外面已经准备好了 captions 和 save_dir
    # create_grid_with_captions(visual_img, captions, save_dir, prompt_text, G)

    def sanitize_for_filename(text: str, max_len: int = 80) -> str:
        # 把文件名里不安全的字符替换掉，避免空格、中文、标点导致问题
        text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
        return text[:max_len]

    caption_base = sanitize_for_filename(prompt_text)

    for i, img in enumerate(visual_img):
        # img: [H, W, 3] uint8, RGB
        pil_img = Image.fromarray(img.astype("uint8"), mode="RGB")

        filename = f"{i:03d}_{caption_base}.png"
        save_path = os.path.join(args.save_dir, filename)

        pil_img.save(save_path)



random.shuffle(prompt_list)
for idx, prompt in enumerate(tqdm(prompt_list, desc="Generating images"), start=1):
    prompt_text = copy.deepcopy(prompt)
    conversation = [
        {
            "role": "User",
            "content": prompt,
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
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