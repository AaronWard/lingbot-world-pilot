"""
server/main.py — interactive LingBot-World-Fast streaming server.

Replaces the old prequant / WanI2V_PreQuant generator_loop entirely. One
persistent WanI2VFastStreaming pipe is created at startup and prewarmed; each
websocket session drives it chunk-by-chunk from live WASD + mouse input.

PROTOCOL (one websocket at /ws):
  client -> server (text JSON):
    {"type":"init","prompt":"...", "image":"<base64 png/jpg>",
     "seed":-1}                         # start a session
    {"type":"input","keys":["w","a"], "dx":12.3,"dy":-4.0,"dt":0.05}
                                        # one input sample (send at your tick rate)
    {"type":"stop"}                     # end session
  server -> client:
    {"type":"ready","width":W,"height":H,"fps":16}      (text JSON)
    {"type":"error","message":"..."}                    (text JSON)
    <binary>  = 12-byte little-endian header
                [uint32 frame_index][uint16 width][uint16 height][uint32 jpeg_len]
                followed by jpeg_len bytes of JPEG.                (binary)

The server batches input samples; whenever enough wall-clock / samples have
accumulated to fill one chunk (chunk_size latent frames == chunk_size*4 pixel
frames), it integrates them into camera poses, calls pipe.step(), and streams
the decoded frames back as JPEGs.

Env config (all optional, sensible defaults):
  LW_CKPT_DIR        path to the base-cam checkpoint dir (must contain T5, VAE,
                     tokenizer, and the lingbot_world_fast/ subfolder)
  LW_DEVICE_ID       CUDA device for the DiT+VAE (default 1 -> the 5090)
  LW_T5_CPU          "1" to keep T5 on CPU (default 1; 4060 is too small for it)
  LW_LOCAL_ATTN      rolling KV window in latent frames (default 12 ~= 3s, ~15GB)
  LW_SINK            pinned origin frames (default 1)
  LW_CHUNK           latent frames per chunk (default 3)
  LW_SHIFT           flow-matching shift (default 3.0 for 480p)
  LW_MAX_AREA        H*W budget (default 480*832)
  LW_QUANT           "" | "fp8" | "nf4"  (DiT quantization; see notes in README)
  LW_HOST/LW_PORT    bind address (default 0.0.0.0:8000)
"""

import asyncio
import base64
import io
import os
import struct
import sys
import time
import uuid

# Reduce fragmentation OOMs: the Fast KV cache + the VAE decode spike sit close
# to the ceiling on a single card. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --- make the upstream `wan` package importable (submodule or sibling clone) ---
LW_REPO = os.environ.get("LW_REPO", os.path.join(os.path.dirname(__file__), "..", "lingbot-world"))
sys.path.insert(0, os.path.abspath(LW_REPO))

from wan.configs import WAN_CONFIGS                       # noqa: E402
from wan.streaming_fast import WanI2VFastStreaming        # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CKPT_DIR     = os.environ.get("LW_CKPT_DIR", "/mnt/data4tb/lingbot-world-base-cam")
DEVICE_ID    = int(os.environ.get("LW_DEVICE_ID", "1"))   # 5090 = cuda:1
T5_CPU       = os.environ.get("LW_T5_CPU", "1") == "1"
LOCAL_ATTN   = int(os.environ.get("LW_LOCAL_ATTN", "8"))  # ~2s window, ~10GB KV
SINK         = int(os.environ.get("LW_SINK", "1"))
CHUNK        = int(os.environ.get("LW_CHUNK", "3"))
SHIFT        = float(os.environ.get("LW_SHIFT", "3.0"))
MAX_AREA     = int(eval(os.environ.get("LW_MAX_AREA", "480*832")))  # noqa: S307 (trusted env)
QUANT        = os.environ.get("LW_QUANT", "").lower()
FAST_SUBFOLDER = os.environ.get("LW_FAST_SUBFOLDER", "lingbot_world_fast")
PREQUANT     = os.environ.get("LW_PREQUANTIZED", "0") == "1"
FPS          = 16

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PIPE: WanI2VFastStreaming | None = None
PIPE_LOCK = asyncio.Lock()           # one session at a time (single GPU)
SESSIONS: dict = {}                  # session_id -> {"prompt", "image", "max_area"}

RES_TO_AREA = {"480p": 480 * 832, "720p": 720 * 1280}


# --------------------------------------------------------------------------- #
# Camera: integrate WASD + mouse into OpenCV camera-to-world poses
# --------------------------------------------------------------------------- #
class Camera:
    """Minimal FPS-style camera in OpenCV convention (x right, y down, z fwd).

    Produces one absolute c2w matrix per *latent* frame. CALIBRATE move_speed /
    look_sensitivity to taste; these set how far the model is asked to move per
    second of input.
    """
    def __init__(self, move_speed=1.0, look_sensitivity=0.0025):
        self.pos = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0     # radians, around world-up (y)
        self.pitch = 0.0
        self.move_speed = move_speed
        self.look_sensitivity = look_sensitivity

    def _rot(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Ry @ Rx

    def _c2w(self):
        R = self._rot()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.pos
        return T

    def integrate(self, keys, dx, dy, dt):
        """Advance the camera by one input sample and return its pose."""
        self.yaw += dx * self.look_sensitivity
        self.pitch = float(np.clip(self.pitch + dy * self.look_sensitivity, -1.3, 1.3))
        R = self._rot()
        fwd = R @ np.array([0, 0, 1.0])
        right = R @ np.array([1.0, 0, 0])
        step = self.move_speed * dt
        k = set(keys or [])
        if "w" in k: self.pos += fwd * step
        if "s" in k: self.pos -= fwd * step
        if "d" in k: self.pos += right * step
        if "a" in k: self.pos -= right * step
        return self._c2w()


def default_intrinsics(width=832, height=480, fov_deg=60.0):
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([f, f, width / 2.0, height / 2.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model load
# --------------------------------------------------------------------------- #
def _install_quant_patch():
    """Make WanModelFast load already-quantized so the full bf16 (~28GB) DiT is
    never materialized on the GPU. NF4 leaves ~7GB for the DiT, which is what
    makes room for the KV cache on a single 32GB card.
    """
    if QUANT == "nf4":
        from diffusers import BitsAndBytesConfig
        from wan.modules.model_fast import WanModelFast
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _orig = WanModelFast.from_pretrained  # bound classmethod (cls already bound)

        def _patched(*args, **kwargs):
            kwargs.setdefault("quantization_config", qcfg)
            return _orig(*args, **kwargs)

        WanModelFast.from_pretrained = staticmethod(_patched)
        print("[lingbot] NF4 quantization armed (load-time)", flush=True)
    elif QUANT == "fp8":
        raise NotImplementedError(
            "fp8 path not wired; use LW_QUANT=nf4 (tested) or leave empty for bf16.")
    elif QUANT not in ("", "bf16"):
        raise ValueError(f"unknown LW_QUANT={QUANT!r}")


@app.on_event("startup")
def load_pipe():
    global PIPE
    cfg = WAN_CONFIGS["i2v-A14B"]
    cfg.fast_noise_checkpoint = FAST_SUBFOLDER       # which DiT subfolder to load
    if PREQUANT:
        # Weights already NF4; the saved config.json carries the quant settings,
        # so we must NOT arm the on-the-fly patch.
        print(f"[lingbot] loading pre-quantized DiT from subfolder '{FAST_SUBFOLDER}'", flush=True)
    else:
        _install_quant_patch()                       # must run BEFORE weights load
    PIPE = WanI2VFastStreaming(
        config=cfg,
        checkpoint_dir=CKPT_DIR,        # 'cam' is auto-detected from this path
        device_id=DEVICE_ID,
        t5_cpu=T5_CPU,
        init_on_cpu=False,
        convert_model_dtype=False,      # let bnb own the dtype when quantizing
        local_attn_size=LOCAL_ATTN,
        sink_size=SINK,
    )
    # Prewarm at the streaming shape so the first frame isn't taxed ~6s.
    dummy = Image.new("RGB", (832, 480))
    PIPE.prewarm(torch.from_numpy(np.array(dummy)).permute(2, 0, 1),
                 max_area=MAX_AREA, frame_num=(CHUNK - 1) * 4 + 1, chunk_size=CHUNK)
    print(f"[lingbot] ready: device=cuda:{DEVICE_ID} t5_cpu={T5_CPU} "
          f"window={LOCAL_ATTN}f (~{LOCAL_ATTN*4/FPS:.1f}s, ~{LOCAL_ATTN*1.28:.0f}GB KV) "
          f"chunk={CHUNK} shift={SHIFT} quant={QUANT or 'bf16'}", flush=True)


def _pack_frame(idx, frame_uint8):
    """frame_uint8: [H,W,3] -> 12-byte header + JPEG bytes."""
    h, w = frame_uint8.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(frame_uint8).save(buf, format="JPEG", quality=85)
    jpg = buf.getvalue()
    header = struct.pack("<IHHI", idx, w, h, len(jpg))
    return header + jpg


# --------------------------------------------------------------------------- #
# Websocket session
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# REST: create a session (matches the frontend's POST /api/session flow)
# --------------------------------------------------------------------------- #
@app.post("/api/session")
async def create_session(
    request: Request,
    prompt: str = Form(""),
    resolution: str = Form("480p"),
    quality: str = Form("balanced"),
    initImage: UploadFile = File(None),
):
    if initImage is None:
        return {"detail": "initImage is required"}
    raw = await initImage.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "prompt": prompt,
        "image": img,
        "max_area": RES_TO_AREA.get(resolution, MAX_AREA),
    }
    host = request.headers.get("host", f"localhost:{os.environ.get('LW_PORT', '8000')}")
    return {"session_id": sid, "ws_url": f"ws://{host}", "ws_path": "/ws"}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    cam = Camera()
    K = default_intrinsics()
    samples: list[tuple] = []     # pending (keys, dx, dy, dt)
    frame_idx = 0
    started = False

    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "text" not in msg or msg["text"] is None:
                continue
            import json
            data = json.loads(msg["text"])
            kind = data.get("type")

            if kind == "init":
                if PIPE_LOCK.locked():
                    await sock.send_text('{"type":"error","message":"GPU busy (single session)"}')
                    continue
                await PIPE_LOCK.acquire()
                sid = data.get("session_id")
                if sid and sid in SESSIONS:
                    sess = SESSIONS.pop(sid)
                    img = sess["image"]
                    prompt = sess["prompt"]
                    sess_area = sess["max_area"]
                elif data.get("image"):
                    img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
                    prompt = data.get("prompt", "")
                    sess_area = MAX_AREA
                else:
                    await sock.send_text('{"type":"error","message":"no session_id or image"}')
                    PIPE_LOCK.release()
                    continue
                seed = int(data.get("seed", -1))
                # start_session is blocking GPU work -> run off the event loop.
                await asyncio.to_thread(
                    PIPE.start_session, img, prompt,
                    max_area=sess_area, chunk_size=CHUNK, shift=SHIFT, seed=seed)
                h = PIPE._s["h"]; w = PIPE._s["w"]
                started = True
                samples.clear()
                frame_idx = 0
                await sock.send_text(f'{{"type":"ready","width":{w},"height":{h},"fps":{FPS}}}')

            elif kind == "input" and started:
                samples.append((data.get("keys", []),
                                float(data.get("dx", 0.0)),
                                float(data.get("dy", 0.0)),
                                float(data.get("dt", 1.0 / FPS))))
                # One chunk == CHUNK latent frames == CHUNK*4 pixel frames of input.
                if len(samples) >= CHUNK * 4:
                    poses = _samples_to_chunk_poses(cam, samples, CHUNK)
                    samples.clear()
                    frames = await asyncio.to_thread(PIPE.step, poses, K)
                    for f in frames:
                        await sock.send_bytes(_pack_frame(frame_idx, f))
                        frame_idx += 1

            elif kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if started:
            await asyncio.to_thread(PIPE.end_session)
        if PIPE_LOCK.locked():
            PIPE_LOCK.release()


def _samples_to_chunk_poses(cam: Camera, samples, chunk_size) -> np.ndarray:
    """Collapse CHUNK*4 input samples into CHUNK latent-frame poses.

    Each latent frame summarizes 4 consecutive input samples; we integrate all
    of them and snapshot the camera pose at each latent-frame boundary.
    """
    poses = []
    per = max(1, len(samples) // chunk_size)
    for i in range(chunk_size):
        group = samples[i * per:(i + 1) * per] or samples[-1:]
        pose = cam._c2w()
        for (keys, dx, dy, dt) in group:
            pose = cam.integrate(keys, dx, dy, dt)
        poses.append(pose)
    return np.stack(poses, axis=0)      # [chunk_size, 4, 4]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("LW_HOST", "0.0.0.0"),
                port=int(os.environ.get("LW_PORT", "8000")))