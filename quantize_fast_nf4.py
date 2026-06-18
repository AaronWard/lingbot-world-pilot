"""
scripts/quantize_fast_nf4.py — quantize the Fast DiT to NF4 ONCE and save it.

Run this a single time. It loads the bf16 Fast DiT, quantizes to NF4 on the
GPU, and writes the 4-bit weights (+ a config.json that records the quant
settings) to  $LW_CKPT_DIR/lingbot_world_fast_nf4 . After that, launch the
server with  LW_FAST_SUBFOLDER=lingbot_world_fast_nf4  LW_PREQUANTIZED=1  and it
loads the small pre-quantized weights directly (~30-60s instead of ~6min, and
no transient 28GB bf16 footprint).

Usage:
    LW_REPO=$PWD/lingbot-world \
    LW_CKPT_DIR=/mnt/data4tb/lingbot-world-base-cam \
    LW_DEVICE_ID=0 \
    python scripts/quantize_fast_nf4.py
"""
import os
import sys

import torch

LW_REPO = os.environ.get("LW_REPO", os.path.join(os.path.dirname(__file__), "..", "lingbot-world"))
sys.path.insert(0, os.path.abspath(LW_REPO))

from diffusers import BitsAndBytesConfig          # noqa: E402
from wan.modules.model_fast import WanModelFast    # noqa: E402

CKPT_DIR = os.environ["LW_CKPT_DIR"]
DEV = int(os.environ.get("LW_DEVICE_ID", "0"))
SRC_SUBFOLDER = os.environ.get("LW_FAST_SUBFOLDER_SRC", "lingbot_world_fast")
OUT_DIR = os.path.join(CKPT_DIR, "lingbot_world_fast_nf4")

print(f">> loading bf16 Fast DiT from {CKPT_DIR}/{SRC_SUBFOLDER} and quantizing to NF4 on cuda:{DEV}")
qcfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = WanModelFast.from_pretrained(
    CKPT_DIR,
    subfolder=SRC_SUBFOLDER,
    torch_dtype=torch.bfloat16,
    quantization_config=qcfg,
    device_map={"": DEV},     # quantize directly on the GPU
)

print(f">> saving NF4 weights to {OUT_DIR}")
model.save_pretrained(OUT_DIR)

print(">> done. Launch the server with:")
print(f"     export LW_FAST_SUBFOLDER=lingbot_world_fast_nf4")
print(f"     export LW_PREQUANTIZED=1")
print(f"     # (LW_QUANT is ignored in this mode — the saved config carries NF4)")
