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
  LW_DEVICE_ID       CUDA device for the DiT / streaming core (default 1 -> the 5090)
  LW_VAE_DEVICE_ID   CUDA device for the VAE (default same as LW_DEVICE_ID)
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
CKPT_DIR      = os.environ.get("LW_CKPT_DIR", "/mnt/data4tb/lingbot-world-base-cam")
DEVICE_ID     = int(os.environ.get("LW_DEVICE_ID", "1"))   # 5090 = cuda:1
VAE_DEVICE_ID = int(os.environ.get("LW_VAE_DEVICE_ID", str(DEVICE_ID)))
T5_CPU        = os.environ.get("LW_T5_CPU", "1") == "1"
LOCAL_ATTN    = int(os.environ.get("LW_LOCAL_ATTN", "8"))  # ~2s window, ~10GB KV
SINK          = int(os.environ.get("LW_SINK", "1"))
CHUNK         = int(os.environ.get("LW_CHUNK", "3"))
SHIFT         = float(os.environ.get("LW_SHIFT", "10.0"))  # i2v-A14B config default; Fast was distilled on this
MAX_AREA      = int(eval(os.environ.get("LW_MAX_AREA", "480*832")))  # noqa: S307 (trusted env)
QUANT         = os.environ.get("LW_QUANT", "").lower()
FAST_SUBFOLDER = os.environ.get("LW_FAST_SUBFOLDER", "lingbot_world_fast")
PREQUANT      = os.environ.get("LW_PREQUANTIZED", "0") == "1"

FPS           = 16
PLAY_FPS = float(os.environ.get("LW_PLAY_FPS", "4"))
# Max queued decoded frames. Prevents runaway RAM use if generation outruns playback.
FRAME_QUEUE_MAX = int(os.environ.get("LW_FRAME_QUEUE_MAX", "64"))

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
    """FPS-style camera in OpenCV convention (x right, y down, z fwd).

    Velocity-based: input messages update `keys` and accumulate mouse delta
    instantly; the generation loop calls advance() once per *latent frame* to
    integrate motion at its own pace. This decouples the (fast) input rate from
    the (slow) generation rate, so there is no backlog and held keys take effect
    on the very next chunk. Tune move_speed / look_sensitivity via env.
    """
    def __init__(self, move_speed=0.3, look_sensitivity=0.0035):
        self.pos = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_speed = move_speed
        self.look_sensitivity = look_sensitivity
        self.keys: set = set()

    def _rot(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Ry @ Rx

    def _c2w(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._rot()
        T[:3, 3] = self.pos
        return T

    def advance(self, dyaw: float, dpitch: float) -> np.ndarray:
        """Advance one latent frame: apply look delta + translate by held keys.
        Translation is per-frame (already scaled by move_speed), so with
        normalize_trans=False downstream the motion is proportional to input."""
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, -1.3, 1.3))
        R = self._rot()
        fwd = R @ np.array([0, 0, 1.0])
        right = R @ np.array([1.0, 0, 0])
        s = self.move_speed
        if "w" in self.keys: self.pos += fwd * s
        if "s" in self.keys: self.pos -= fwd * s
        if "d" in self.keys: self.pos += right * s
        if "a" in self.keys: self.pos -= right * s
        return self._c2w()


MOVE_SPEED = float(os.environ.get("LW_MOVE_SPEED", "0.3"))
LOOK_SENS  = float(os.environ.get("LW_LOOK_SENS", "0.0035"))


def default_intrinsics(width=832, height=480, fov_deg=60.0):
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([f, f, width / 2.0, height / 2.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model load
# --------------------------------------------------------------------------- #
def _install_quant_patch():
    """Make WanModelFast load quantized. NF4 (bitsandbytes) -> ~7GB on GPU,
    quantized shard-by-shard during load (peak ~17GB), slow matmul, max window.
    FP8 (native torch._scaled_mm, no torchao) -> loads the bf16 weights on CPU
    (~28GB RAM transient), swaps the block Linears to fp8 there, then the pipe
    moves the ~14GB fp8 model to GPU. Blackwell fp8 tensor cores: faster +
    cleaner than NF4, slightly smaller KV window.
    """
    from wan.modules.model_fast import WanModelFast

    if QUANT == "nf4":
        from diffusers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = None
        _orig = WanModelFast.from_pretrained

        def _patched_nf4(*args, **kwargs):
            kwargs.setdefault("quantization_config", qcfg)
            return _orig(*args, **kwargs)

        WanModelFast.from_pretrained = staticmethod(_patched_nf4)
        print("[lingbot] NF4 quantization armed (load-time)", flush=True)

    elif QUANT == "fp8":
        # Native torch._scaled_mm path (no torchao): load bf16 on CPU, swap the
        # block Linears to fp8 there, then the pipe moves the ~14GB model to GPU.
        from wan.fp8_linear import fp8_supported, swap_to_fp8
        if not fp8_supported(DEVICE_ID):
            raise RuntimeError(
                "torch._scaled_mm fp8 not usable on this build — use LW_QUANT=nf4.")
        _orig = WanModelFast.from_pretrained

        def _patched_fp8(*args, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", True)   # load bf16 on CPU
            model = _orig(*args, **kwargs)
            count = swap_to_fp8(model)                     # in-place, on CPU
            print(f"[lingbot] FP8: swapped {count} block Linears via _scaled_mm", flush=True)
            return model

        WanModelFast.from_pretrained = staticmethod(_patched_fp8)
        print("[lingbot] FP8 quantization armed (native _scaled_mm)", flush=True)

    elif QUANT in ("", "bf16"):
        return
    else:
        raise ValueError(f"unknown LW_QUANT={QUANT!r}")

def _move_pipe_vae_to_device(pipe: WanI2VFastStreaming, vae_device_id: int) -> torch.device:
    """Move the VAE's underlying torch module/state to a separate CUDA device.

    Wan2_1_VAE is a wrapper, not necessarily an nn.Module, so pipe.vae.to(...)
    may not exist. This moves the wrapper's torch modules/tensors, including
    nested tensor containers like self.scale, and wraps encode/decode so tensors
    cross the DiT<->VAE device boundary explicitly.
    """
    vae_device = torch.device(f"cuda:{vae_device_id}")
    dit_device = torch.device(f"cuda:{DEVICE_ID}")

    vae = getattr(pipe, "vae", None)

    if vae is None and hasattr(pipe, "model"):
        vae = getattr(pipe.model, "vae", None)

    if vae is None:
        print("[lingbot] WARNING: could not find VAE object to move", flush=True)
        return vae_device

    if vae_device_id == DEVICE_ID:
        print(f"[lingbot] VAE staying on cuda:{DEVICE_ID}", flush=True)
        return vae_device

    moved = []

    def move_obj(obj, device):
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
            return obj

        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)

        if isinstance(obj, list):
            return [move_obj(x, device) for x in obj]

        if isinstance(obj, tuple):
            return tuple(move_obj(x, device) for x in obj)

        if isinstance(obj, dict):
            return {k: move_obj(v, device) for k, v in obj.items()}

        return obj

    def contains_torch_obj(obj):
        if isinstance(obj, torch.nn.Module) or torch.is_tensor(obj):
            return True

        if isinstance(obj, (list, tuple)):
            return any(contains_torch_obj(x) for x in obj)

        if isinstance(obj, dict):
            return any(contains_torch_obj(v) for v in obj.values())

        return False

    if isinstance(vae, torch.nn.Module):
        vae.to(vae_device)
        moved.append(type(vae).__name__)
    else:
        # Wan2_1_VAE-style wrapper: move modules/tensors stored on the wrapper,
        # including nested containers such as self.scale = [mean, std].
        for name, value in vars(vae).items():
            if contains_torch_obj(value):
                setattr(vae, name, move_obj(value, vae_device))
                moved.append(name)

    if moved:
        print(f"[lingbot] moved VAE internals to {vae_device}: {', '.join(moved)}", flush=True)
    else:
        print(
            f"[lingbot] WARNING: VAE object has no direct torch modules/tensors to move: "
            f"{type(vae).__name__}",
            flush=True,
        )

    orig_encode = vae.encode
    orig_decode = vae.decode

    def encode_on_vae_device(xs, *args, **kwargs):
        xs = move_obj(xs, vae_device)
        args = move_obj(args, vae_device)
        kwargs = move_obj(kwargs, vae_device)

        with torch.cuda.device(vae_device):
            ys = orig_encode(xs, *args, **kwargs)

        # streaming_fast expects latents back on the DiT / streaming device.
        return move_obj(ys, dit_device)

    def decode_on_vae_device(zs, *args, **kwargs):
        zs = move_obj(zs, vae_device)
        args = move_obj(args, vae_device)
        kwargs = move_obj(kwargs, vae_device)

        with torch.cuda.device(vae_device):
            frames = orig_decode(zs, *args, **kwargs)

        # Keep downstream behavior compatible with streaming_fast.
        return move_obj(frames, dit_device)

    vae.encode = encode_on_vae_device
    vae.decode = decode_on_vae_device

    print(
        f"[lingbot] patched VAE encode/decode device bridge: "
        f"dit={dit_device} <-> vae={vae_device}",
        flush=True,
    )

    return vae_device


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

    _move_pipe_vae_to_device(PIPE, VAE_DEVICE_ID)

    # Prewarm at the streaming shape so the first frame isn't taxed ~6s.
    dummy = Image.new("RGB", (832, 480))
    PIPE.prewarm(torch.from_numpy(np.array(dummy)).permute(2, 0, 1),
                 max_area=MAX_AREA, frame_num=(CHUNK - 1) * 4 + 1, chunk_size=CHUNK)
    print(f"[lingbot] ready: device=cuda:{DEVICE_ID} vae=cuda:{VAE_DEVICE_ID} t5_cpu={T5_CPU} "
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
    import json
    cam = Camera(move_speed=MOVE_SPEED, look_sensitivity=LOOK_SENS)
    K = default_intrinsics()
    mouse = {"dx": 0.0, "dy": 0.0}
    state = {"running": False, "frame_idx": 0}
    gen_task = None

    async def gen_loop():
        """Generate continuously: each iteration advances the camera by held
        input over CHUNK latent frames, runs one step, streams the frames.
        Decoupled from input arrival, so WASD affects the next chunk immediately
        and there is no backlog."""
        try:
            while state["running"]:
                dx = mouse["dx"]; dy = mouse["dy"]
                mouse["dx"] = 0.0; mouse["dy"] = 0.0
                dyaw = (dx * cam.look_sensitivity) / CHUNK
                dpitch = (dy * cam.look_sensitivity) / CHUNK
                poses = np.stack([cam.advance(dyaw, dpitch) for _ in range(CHUNK)], axis=0)
                frames = await asyncio.to_thread(PIPE.step, poses, K)
                for f in frames:
                    await sock.send_bytes(_pack_frame(state["frame_idx"], f))
                    state["frame_idx"] += 1
        except Exception as e:               # surface, don't die silently
            state["running"] = False
            try:
                await sock.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
            print(f"[lingbot] gen_loop error: {e!r}", flush=True)

    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "text" not in msg or msg["text"] is None:
                continue
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
                    img = sess["image"]; prompt = sess["prompt"]; sess_area = sess["max_area"]
                elif data.get("image"):
                    img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
                    prompt = data.get("prompt", ""); sess_area = MAX_AREA
                else:
                    await sock.send_text('{"type":"error","message":"no session_id or image"}')
                    PIPE_LOCK.release()
                    continue
                seed = int(data.get("seed", -1))
                await asyncio.to_thread(
                    PIPE.start_session, img, prompt,
                    max_area=sess_area, chunk_size=CHUNK, shift=SHIFT, seed=seed,
                    normalize_trans=False)        # raw, proportional camera motion
                h = PIPE._s["h"]; w = PIPE._s["w"]
                state["running"] = True
                state["frame_idx"] = 0
                await sock.send_text(f'{{"type":"ready","width":{w},"height":{h},"fps":{FPS}}}')
                gen_task = asyncio.create_task(gen_loop())

            elif kind == "input" and state["running"]:
                cam.keys = set(data.get("keys", []))      # live: takes effect next chunk
                mouse["dx"] += float(data.get("dx", 0.0))
                mouse["dy"] += float(data.get("dy", 0.0))

            elif kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        state["running"] = False
        if gen_task is not None:
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):
                pass
        if PIPE.__class__ and getattr(PIPE, "_s", None) is not None:
            await asyncio.to_thread(PIPE.end_session)
        if PIPE_LOCK.locked():
            PIPE_LOCK.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("LW_HOST", "0.0.0.0"),
                port=int(os.environ.get("LW_PORT", "8000")))