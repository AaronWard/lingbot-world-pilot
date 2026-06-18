#!/usr/bin/env bash
# scripts/download_fast.sh
#
# Assemble the checkpoint dir the Fast pipeline expects. You are running the
# FAST causal DiT, but it shares the T5 text encoder, the VAE, and the
# tokenizer with the base-cam release (those are NOT in the fast repo). So you
# pull a FEW files from base-cam (NOT its ~56GB high/low-noise DiT) plus the
# Fast DiT.
#
# Final layout (everything on /mnt/data4tb):
#   $CKPT_DIR/
#     models_t5_umt5-xxl-enc-bf16.pth   <- base-cam (T5 encoder, ~11GB)
#     Wan2.1_VAE.pth                    <- base-cam (VAE)
#     google/umt5-xxl/                  <- base-cam (tokenizer)
#     lingbot_world_fast/               <- the Fast causal DiT (this is what you run)
#
# The dir name MUST contain "cam" (and not "act"): the pipeline auto-detects
# control_type from the path (`if 'cam' in checkpoint_dir: 'cam'`). That's the
# only reason the word base-cam appears — you are not running the base DiT.
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/mnt/data4tb/lingbot-world-base-cam}"
mkdir -p "$CKPT_DIR"
command -v hf >/dev/null || pip install -U "huggingface_hub>=0.34,<1.0" >/dev/null

echo ">> shared support files from base-cam (T5 + VAE + tokenizer only)"
hf download robbyant/lingbot-world-base-cam \
  --local-dir "$CKPT_DIR" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth" "google/**" "configuration.json"

echo ">> LingBot-World-Fast DiT -> $CKPT_DIR/lingbot_world_fast"
hf download robbyant/lingbot-world-fast \
  --local-dir "$CKPT_DIR/lingbot_world_fast"

echo ">> done. Launch with LW_CKPT_DIR=$CKPT_DIR"
du -sh "$CKPT_DIR" "$CKPT_DIR/lingbot_world_fast" 2>/dev/null || true
ls -la "$CKPT_DIR"