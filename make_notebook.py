import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Markdown Title
cells.append(nbf.v4.new_markdown_cell("# Train SDXL LoRA for Bonnie (West Highland White Terrier)"))

# Module 1: Environment & Accelerated Dependencies
cell_1_code = """!pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q -U accelerate transformers diffusers peft bitsandbytes xformers scikit-image opencv-python pillow nbformat datasets

import os
os.environ["ACCELERATE_USE_FP16"] = "true"
os.environ["ACCELERATE_USE_XFORMERS"] = "true"
os.environ["ACCELERATE_GRADIENT_CHECKPOINTING"] = "true"
"""
cells.append(nbf.v4.new_code_cell(cell_1_code))

cell_2_code = """import accelerate
from accelerate.utils import write_basic_config
import os

# Write a basic accelerate config for single GPU, fp16 mixed precision
config_path = os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml')
os.makedirs(os.path.dirname(config_path), exist_ok=True)
config_content = '''compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
'''
with open(config_path, 'w') as f:
    f.write(config_content.strip())

print("Accelerate config created for single GPU, fp16 mixed precision.")
"""
cells.append(nbf.v4.new_code_cell(cell_2_code))

# Module 2: Image Quality Auditing, Upscaling & Preprocessing
cell_3_code = """import os
import cv2
import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask
import shutil

input_dir = "dataset/raw"
output_dir = "dataset/processed"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Bucketing aspect ratios for SDXL
BUCKETS = [
    (1024, 1024),
    (896, 1152),
    (1152, 896)
]

def closest_bucket(w, h):
    aspect_ratio = w / h
    best_bucket = min(BUCKETS, key=lambda b: abs((b[0] / b[1]) - aspect_ratio))
    return best_bucket

def process_image(img_path, save_path):
    img = Image.open(img_path)

    # 3. Color Space & Alpha Sanitization
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 1. Resolution & Super-Resolution Check
    if min(w, h) < 1024:
        scale = 1024.0 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Unsharp masking for enhancement after upscaling
        img_np = unsharp_mask(img_np, radius=1.0, amount=1.0, channel_axis=-1)
        img_np = (img_np * 255).astype(np.uint8)
        w, h = new_w, new_h

    # 2. Aspect-Ratio Bucketing
    target_w, target_h = closest_bucket(w, h)

    img_pil = Image.fromarray(img_np)

    scale_w = target_w / w
    scale_h = target_h / h
    scale = max(scale_w, scale_h)

    resize_w, resize_h = int(w * scale), int(h * scale)
    img_pil = img_pil.resize((resize_w, resize_h), Image.BICUBIC)

    left = (resize_w - target_w) / 2
    top = (resize_h - target_h) / 2
    right = (resize_w + target_w) / 2
    bottom = (resize_h + target_h) / 2

    img_pil = img_pil.crop((left, top, right, bottom))
    img_pil.save(save_path, format="JPEG", quality=95)
    print(f"Processed {os.path.basename(img_path)} -> {target_w}x{target_h}")

# In Kaggle, users will upload images to dataset/raw.
# We simulate calling this by checking if the directory has files.
if os.path.exists(input_dir):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) == 0:
        print(f"Warning: No images found in {input_dir}. Please upload exactly 10 images of Bonnie.")
    for f in files:
        process_image(os.path.join(input_dir, f), os.path.join(output_dir, f))
else:
    print(f"Warning: {input_dir} does not exist. Create it and upload images.")

print("Preprocessing pipeline initialized.")
"""
cells.append(nbf.v4.new_code_cell(cell_3_code))

# Module 3: Anti-Overfitting Captioning & Feature Decoupling
cell_4_code = """import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "microsoft/Florence-2-base"

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

processed_dir = "dataset/processed"
metadata_path = "dataset/processed/metadata.jsonl"
metadata = []

redundant_terms = ["a photograph of a dog", "a picture of a dog", "a photo of a dog", "a dog", "photograph of a dog", "picture of a dog", "photo of a dog"]

def clean_caption(text):
    text = text.lower()
    for term in redundant_terms:
        text = text.replace(term, "")
    text = " ".join(text.split())
    if text.startswith(','): text = text[1:].strip()
    return text

if os.path.exists(processed_dir):
    files = [f for f in os.listdir(processed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) == 0:
        print(f"Warning: No images found in {processed_dir}. Run preprocessing first.")

    for fname in files:
        img_path = os.path.join(processed_dir, fname)
        image = Image.open(img_path)

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))

        raw_caption = parsed_answer["<MORE_DETAILED_CAPTION>"]
        cleaned_caption = clean_caption(raw_caption)

        final_caption = f"bonnie_dog, {cleaned_caption}".strip(', ')

        # Format required for diffusers dreambooth metadata
        metadata.append({"file_name": fname, "text": final_caption})
        print(f"{fname}: {final_caption}")

    if metadata:
        with open(metadata_path, "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\\n")
        print(f"Saved metadata to {metadata_path}")
else:
    print("Processed directory not found. Please run preprocessing step first.")

# Cleanup VRAM
del model
torch.cuda.empty_cache()
"""
cells.append(nbf.v4.new_code_cell(cell_4_code))


# Module 4: Prior Preservation (Class Regularization)
cell_5_code = """import os
import torch
from diffusers import DiffusionPipeline

class_dir = "dataset/class_images"
os.makedirs(class_dir, exist_ok=True)

num_class_images = 60
class_prompt = "a photo of a west highland white terrier dog"

print(f"Generating {num_class_images} class images...")

torch.cuda.empty_cache()
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

batch_size = 4
for i in range(0, num_class_images, batch_size):
    current_batch_size = min(batch_size, num_class_images - i)
    images = pipeline(
        prompt=[class_prompt] * current_batch_size,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images

    for j, img in enumerate(images):
        idx = i + j
        img.save(os.path.join(class_dir, f"class_{idx:03d}.jpg"))
        print(f"Saved class image {idx + 1}/{num_class_images}")

del pipeline
torch.cuda.empty_cache()
"""
cells.append(nbf.v4.new_code_cell(cell_5_code))

# Module 5: Training Hyperparameters & Deep Regularization Setup
cell_6_code = """import urllib.request
import os
import subprocess

trainer_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py"
trainer_script = "train_dreambooth_lora_sdxl.py"

if not os.path.exists(trainer_script):
    print("Downloading trainer script...")
    urllib.request.urlretrieve(trainer_url, trainer_script)
    print("Downloaded train_dreambooth_lora_sdxl.py")

    # Patch script to explicitly use lora_alpha=16 instead of lora_alpha=rank
    subprocess.run(["sed", "-i", "s/lora_alpha=args.rank/lora_alpha=16/g", trainer_script])
    print("Patched trainer script for lora_alpha=16")
else:
    print("Trainer script already exists.")
"""
cells.append(nbf.v4.new_code_cell(cell_6_code))


cell_7_code = """import os

# Create training execution script
command = '''accelerate launch \\
  train_dreambooth_lora_sdxl.py \\
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
  --dataset_name="dataset/processed" \\
  --caption_column="text" \\
  --instance_prompt="bonnie_dog" \\
  --with_prior_preservation \\
  --prior_loss_weight=1.0 \\
  --class_data_dir="dataset/class_images" \\
  --class_prompt="a photo of a west highland white terrier dog" \\
  --resolution=1024 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --gradient_checkpointing \\
  --learning_rate=1e-4 \\
  --lr_scheduler="cosine_with_restarts" \\
  --lr_warmup_steps=50 \\
  --max_train_steps=1200 \\
  --checkpointing_steps=300 \\
  --seed=42 \\
  --output_dir="bonnie-lora-sdxl" \\
  --mixed_precision="fp16" \\
  --use_8bit_adam \\
  --rank=32 \\
  --snr_gamma=5.0 \\
  --noise_offset=0.05 \\
  --report_to="tensorboard"
'''

with open("run_training.sh", "w") as f:
    f.write(command)

print("Starting training... (This may take several hours)")
!bash run_training.sh
"""
cells.append(nbf.v4.new_code_cell(cell_7_code))


# Module 6: Model Validation & Checkpoint Evaluation
cell_8_code = """import os
import torch
import numpy as np
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt

output_dir = "bonnie-lora-sdxl"
checkpoints = [f"checkpoint-{step}" for step in [300, 600, 900, 1200]]
prompts = [
    "a sharp portrait of bonnie_dog, studio lighting",
    "bonnie_dog running on a snow-covered mountain, cinematic lighting",
    "bonnie_dog wearing sunglasses in a vintage car"
]
fixed_seed = 42

print("Loading base model and VAE...")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")
pipe.enable_model_cpu_offload()

results = {ckpt: [] for ckpt in checkpoints}

for ckpt in checkpoints:
    ckpt_path = os.path.join(output_dir, ckpt)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt} not found. Skipping.")
        continue

    pipe.load_lora_weights(ckpt_path)
    print(f"Generating images for {ckpt}...")

    for prompt in prompts:
        generator = torch.Generator(device="cuda").manual_seed(fixed_seed)
        image = pipe(
            prompt,
            num_inference_steps=30,
            generator=generator,
            guidance_scale=7.5
        ).images[0]
        results[ckpt].append(image)

    pipe.unload_lora_weights()

# Plot grid
num_prompts = len(prompts)
valid_ckpts = [c for c in checkpoints if os.path.exists(os.path.join(output_dir, c))]
num_ckpts = len(valid_ckpts)

if num_ckpts > 0:
    fig, axes = plt.subplots(num_prompts, num_ckpts, figsize=(5 * num_ckpts, 5 * num_prompts))

    if num_prompts == 1 and num_ckpts == 1:
        axes = np.array([[axes]])
    elif num_prompts == 1:
        axes = axes[None, :]
    elif num_ckpts == 1:
        axes = axes[:, None]

    for col, ckpt in enumerate(valid_ckpts):
        for row, image in enumerate(results[ckpt]):
            ax = axes[row, col]
            ax.imshow(image)
            ax.axis('off')
            if row == 0:
                ax.set_title(f"{ckpt}")
            if col == 0:
                ax.text(-0.1, 0.5, prompts[row][:30]+"...", transform=ax.transAxes,
                        rotation=90, va='center', ha='right', fontsize=12)

    plt.tight_layout()
    plt.savefig("validation_grid.png")
    plt.show()
else:
    print("No checkpoints evaluated.")
"""
cells.append(nbf.v4.new_code_cell(cell_8_code))

nb.cells = cells
with open("train_bonnie_sdxl_lora.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook train_bonnie_sdxl_lora.ipynb successfully created.")
