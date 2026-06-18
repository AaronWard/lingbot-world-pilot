"""
scripts/quantize_fast_fp8.py — quantize the Fast DiT to FP8 ONCE and save it.

FP8 (Blackwell-native) is faster than NF4 (real fp8 tensor cores, no dequant
tax) and higher precision, at ~14GB vs NF4's ~7GB. Run once, then launch with
  LW_FAST_SUBFOLDER=lingbot_world_fast_fp8  LW_PREQUANTIZED=1 .

Needs a recent torchao matching your torch:  pip install -U torchao

NOTE: torchao serialization is less battle-tested than bitsandbytes. If the
SAVE here or the RELOAD in the server errors, skip this script and use the
on-the-fly path instead (no pre-quant, ~6min load each launch):
    LW_FAST_SUBFOLDER=lingbot_world_fast  LW_PREQUANTIZED=0  LW_QUANT=fp8

Usage:
    LW_REPO=$PWD/lingbot-world \
    LW_CKPT_DIR=/mnt/data4tb/lingbot-world-base-cam \
    LW_DEVICE_ID=0 \
    python scripts/quantize_fast_fp8.py
"""
import os
import sys

import torch

LW_REPO = os.environ.get("LW_REPO", os.path.join(os.path.dirname(__file__), "..", "lingbot-world"))
sys.path.insert(0, os.path.abspath(LW_REPO))
import logging

from diffusers import TorchAoConfig                # noqa: E402
from wan.modules.model_fast import WanModelFast    # noqa: E402

CKPT_DIR = os.environ["LW_CKPT_DIR"]
DEV = int(os.environ.get("LW_DEVICE_ID", "0"))
SRC = os.environ.get("LW_FAST_SUBFOLDER_SRC", "lingbot_world_fast")
QUANT_TYPE = os.environ.get("LW_FP8_QUANT", "float8dq_e4m3")  # or "float8wo" (robust fallback)
OUT_DIR = os.path.join(CKPT_DIR, "lingbot_world_fast_fp8")

print(f">> loading bf16 Fast DiT from {CKPT_DIR}/{SRC}, quantizing to {QUANT_TYPE} on cuda:{DEV}")
qcfg = TorchAoConfig(QUANT_TYPE)
model = WanModelFast.from_pretrained(
    CKPT_DIR,
    subfolder=SRC,
    torch_dtype=torch.bfloat16,
    quantization_config=qcfg,
    device_map={"": DEV},
)

print(f">> saving FP8 weights to {OUT_DIR}")
# torchao tensors aren't safetensors-serializable -> must use safe_serialization=False
model.save_pretrained(OUT_DIR, safe_serialization=False)

print(">> done. Launch the server with:")
print("     export LW_FAST_SUBFOLDER=lingbot_world_fast_fp8")
print("     export LW_PREQUANTIZED=1")
print("     export LW_LOCAL_ATTN=8     # fp8 leaves a bit less room than nf4")
