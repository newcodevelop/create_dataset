import os
import torch

from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"

INPUT_FOLDER  = "/workspace/ocr_benchmark/v7_hi_indicxlit/images"
OUTPUT_FOLDER = "/workspace/ocr_benchmark/v7_hi_indicxlit/realistic_images"

# Images with longest side above this will be resized down.
# Tune based on your VRAM:
#   24 GB -> 1024
#   16 GB -> 768
#   8 GB  -> 512
MAX_SIZE = 1024

# =====================================================
# GPU CHECK
# =====================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Check your GPU access before running.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =====================================================
# PROMPT
# =====================================================

prompt = """Transform this clean receipt image into a realistic scanned thermal-paper receipt photo.

The input image already contains ALL the text that should appear. Your job is ONLY to add realistic
physical texture and scanning artifacts — nothing else.

STRICT RULES:
- Do NOT add any new text, words, stamps, watermarks, or labels of any kind.
- Do NOT add words like "IMPORTANT", "IMPORTANE", "COPY", "VOID", or anything similar.
- Do NOT add any text that is not already visible in the input image.
- Preserve every character exactly as shown — Hindi (Devanagari) and English.
- Preserve all prices, dates, quantities, totals, and layout exactly.

ONLY add these visual effects:
- thermal printer output look on thermal paper
- slight paper wrinkles and subtle shadows
- faded thermal ink texture
- realistic receipt paper grain
- mild scanner artifacts and noise

Keep the receipt highly OCR readable."""

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted text, missing text, "
    "watermark, stamp, label, banner, badge, overlay text, "
    "IMPORTANT, IMPORTANE, COPY, VOID, border box, extra text"
)

# =====================================================
# HELPERS
# =====================================================

def resize_if_needed(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img  # small enough, keep original
    scale = max_size / max(w, h)
    new_w = (int(w * scale) // 8) * 8
    new_h = (int(h * scale) // 8) * 8
    print(f"  Image too large ({w}x{h}), resizing to {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading Qwen-Image-Edit-2511...")

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

device = next(pipeline.transformer.parameters()).device
print(f"Transformer is on: {device}\n")

# =====================================================
# PROCESS ALL IMAGES IN FOLDER
# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

total = len(image_files)
print(f"Found {total} images in '{INPUT_FOLDER}'")

already_done = set(os.listdir(OUTPUT_FOLDER))
pending = [f for f in image_files if f not in already_done]
skipped = total - len(pending)

if skipped:
    print(f"Skipping {skipped} already processed image(s). {len(pending)} remaining.\n")

for idx, fname in enumerate(pending, start=skipped + 1):
    input_path  = os.path.join(INPUT_FOLDER, fname)
    output_path = os.path.join(OUTPUT_FOLDER, fname)

    print(f"[{idx}/{total}] Processing: {fname} ...")

    receipt_img = Image.open(input_path).convert("RGB")
    receipt_img = resize_if_needed(receipt_img)

    with torch.inference_mode():
        result = pipeline(
            image=[receipt_img],
            prompt=prompt,
            height=receipt_img.height,
            width=receipt_img.width,
            num_inference_steps=40,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            negative_prompt=NEGATIVE_PROMPT,
            generator=torch.manual_seed(42),
            num_images_per_prompt=1,
        )

    output = result.images[0]
    output.save(output_path)
    print(f"[{idx}/{total}] Saved: {output_path}")

print(f"\nDone! {total} images total ({skipped} were already done, {len(pending)} processed now).")