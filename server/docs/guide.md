
Got it. I’ll treat this as a **code review + “what to fix next”** pass, assuming the goal is: **your React app becomes a real client** for this FastAPI backend, streaming frames + sending WASD/mouse input with low-feel latency.

## 1) Code understanding (what you have now)

### Backend (server/main.py)

You’ve implemented the right overall shape:

* **POST `/api/session`**
  Accepts `prompt`, `resolution`, `quality`, optional `initImage`, creates `SessionState`, returns `session_id` + `ws_url`.

* **WS `/ws/session/{session_id}`** (duplex)

  * Background `generator_loop()` produces **short chunks** (default 9 frames @ 16fps target) using:

    * rolling `init_img` = **last generated frame** (1-frame continuation)
    * `poses.npy` + `intrinsics.npy` written into a per-session work directory
    * `pipeline.generate(prompt, init_img, action_path, max_area, frame_num, steps, guide, seed)`
  * Enqueues each frame as **JPEG bytes** + JSON header into `frame_queue`
  * Sends:

    * **binary frames**: `[4-byte header_len][header_json][jpeg_bytes]`
    * **telemetry**: JSON text messages periodically

* You’ve got **backpressure + dropping**: queue maxsize, drop oldest if full, low/high watermarks.

### Frontend (current state)

Frontend is still a **mock renderer** (grid + fake latency), but it already has:

* input capture (WASD/mouse, pointer lock)
* telemetry HUD plumbing
* the conceptual loop structure that we’ll replace with:

  * create session via HTTP
  * connect WS
  * send input messages
  * receive binary frames + telemetry

So the overall architecture matches what you want.

---

## 2) Spec + plan of attack (what must be true before we write any JS)

### 2.1 Wire protocol: lock it down

You’re already close. I’d formalize these as the “contract”:

#### HTTP: create session

`POST /api/session` with `multipart/form-data`

Fields:

* `prompt: string` (required)
* `resolution: "480p" | "720p"` (default "480p")
* `quality: "latency" | "balanced" | "quality"` (default "balanced")
* `initImage: file?` (optional)

Response (JSON):

```json
{
  "session_id": "uuid",
  "ws_url": "ws://HOST:8000/ws/session/uuid",
  "resolution": "480p",
  "quality": "balanced"
}
```

#### WebSocket: client → server input

Text JSON:

```json
{
  "type": "input",
  "seq": 123,
  "client_ts_ms": 1700000000000,
  "state": { "w":true, "a":false, "s":false, "d":false, "space":false, "mouseX":3.2, "mouseY":-1.1 }
}
```

#### WebSocket: server → client telemetry

Text JSON:

```json
{
  "type": "telemetry",
  "server_ts_ms": 1700000000123,
  "fps": 14.7,
  "bufferMs": 820,
  "latencyMs": 210,
  "generationTimeMs": 610
}
```

#### WebSocket: server → client frame

Binary:

```
u32_le header_len
header JSON bytes
jpeg bytes
```

Header JSON **should include enough info for real latency accounting**:

```json
{
  "type": "frame",
  "session_id": "...",
  "frame_id": 42,
  "w": 832,
  "h": 480,
  "format": "jpeg",
  "server_ts_ms": 1700000000456,

  "input_seq": 123,
  "input_client_ts_ms": 1700000000000,
  "chunk_id": 7,
  "chunk_frame_idx": 3
}
```

> Right now you don’t include `input_seq` / `chunk_id` / `chunk_frame_idx`, so you can’t measure true “input → displayed frame” latency. We should add those before updating JS.

---

### 2.2 The biggest “correctness” gaps to fix in backend **now**

These are the important ones, in order.

#### Blocker A — repo layout / import path ambiguity

`from generate_prequant import WanI2V_PreQuant` assumes `generate_prequant.py` exists in `REPO_ROOT`.

But your directory listing does **not** include it, meaning one of these is true:

* you copied your webapp into the NF4 repo root (fine), **or**
* this backend will crash on import (likely)

✅ Fix: make the model repo path explicit via env var:

* `LINGBOT_MODEL_REPO=/path/to/lingbot-world-base-cam-nf4`
* add that to `sys.path`
* use that as `checkpoint_dir`

This makes your webapp repo independent.

#### Blocker B — “no image provided => generate first frame”

Right now you create a neutral gray image. That’s **not** what you described.

Reality check: **LingBot-World base-cam is i2v**. If you want true “no image,” you need a **text-to-image** step (separate model) or a baked-in initial frame generator.

✅ Fix options:

* **MVP**: require init image for now (UI can enforce it).
* **Better**: integrate a small T2I (SDXL-turbo / etc.) to produce init image if missing.
* **Hack**: generate 1–2 frames from gray and use the first as init. (Works but quality will be “random vibe.”)

We should decide which you want before wiring the frontend.

#### Important C — FastAPI error handling

You use `raise ValueError` / `RuntimeError`. That will return 500s.

✅ Fix: use `HTTPException(status_code=400/409, detail=...)`.

#### Important D — ws_url is hardcoded to localhost

`ws_url=f"ws://localhost:8000/..."`
If the client is on a different device, or you access via LAN hostname, this breaks.

✅ Fix: either:

* return relative `"/ws/session/{id}"` and let frontend build URL, or
* derive host from request headers (more annoying behind proxies).

#### Important E — telemetry “latencyMs” isn’t meaningful yet

You compute `now - latest_input_ts`. That’s not “latency,” it’s “time since last input was received”.

✅ Fix: tag each frame with `input_seq`/`input_ts` used for that chunk; compute input→frame latency off that.

---

### 2.3 Plan of attack (tight sequence)

1. **Harden backend contract** (add missing header fields, fix HTTP errors, fix ws_url, fix model repo path).
2. Add **a quick CLI test client** (Python or Node) to prove:

   * session create works
   * websocket streams frames
   * jpeg frames decode
3. Only then: replace the frontend mock loop with real WS code.

---

## 3) Backend fixes I would do immediately (before any JS)

Below are the exact changes I’d recommend (conceptually). If you want, I can paste a cleaned-up full `server/main.py` next—but this is what must change first.

### 3.1 Make model repo path explicit

Add at top:

* `MODEL_REPO = Path(os.getenv("LINGBOT_MODEL_REPO", REPO_ROOT))`
* import `generate_prequant` from `MODEL_REPO`
* pass `checkpoint_dir=str(MODEL_REPO)`

This eliminates brittle “must live inside NF4 repo” coupling.

### 3.2 Replace exceptions with HTTPException

Use:

* 400 for invalid params
* 409 for “session limit reached”

### 3.3 Frame header: add chunk + input metadata

In `generator_loop`:

* Keep `st.latest_input_seq` and `st.latest_input_client_ts_ms` (store from InputMsg)
* Add `chunk_id` counter
* When producing frames, stamp:

  * `input_seq`, `input_client_ts_ms`
  * `chunk_id`, `chunk_frame_idx`

### 3.4 Decide behavior when initImage is missing

Pick one:

* enforce init image required (clean)
* or add T2I dependency (more complex)
* or do “gray bootstrap” intentionally and document it

Right now you’re doing bootstrap but not calling it that.

---

## 4) One more subtle point: “ever expanding window”

Your current “continuation” is **last-frame only**:

* `init_img = last frame of previous chunk`

That’s a **1-frame context**, not a growing window.
If LingBot’s continuation pathway supports video conditioning (N previous frames), you’ll want:

* a ring buffer of last K frames (e.g., 16–32)
* pass those as video context if the pipeline supports it

This is the difference between:

* “feels like a coherent world for longer”
  vs
* “drifts quickly / forgets”

Before implementing, we should inspect what `WanI2V_PreQuant.generate()` supports (image-only vs video-to-video). Your current code assumes image-only, which is fine for MVP.

---

## What I need from you to proceed cleanly

Just one thing (no personal info): **is this webapp repo actually inside the NF4 repo root**, or is it separate?

* If separate: we must implement `LINGBOT_MODEL_REPO` as above (strongly recommended anyway).
* If inside: it’ll “work” as-is but still brittle long-term.

Either way, next step on my side would be: **deliver an updated backend `main.py` that locks the protocol**, and a tiny test client to validate streaming. Then we move to JS.

If you say “separate repo,” I’ll assume that going forward and write the backend accordingly.


---
Below is a **fully explicit**, end-to-end set of steps + **full backend code** that will run on your GPU machine, with:

* your app cloned into: `/home/aw/Documents/github/_homelab/`
* models cloned into: `/home/aw/Documents/models/lingbot`
* GPU pinned via: `CUDA_VISIBLE_DEVICES=GPU-11481043-00bd-5b3f-02e7-3138b3f915be`
* FastAPI backend that:

  * creates sessions over HTTP
  * streams frames over WebSocket (binary JPEG packets)
  * accepts WASD/mouse input messages
  * tags each frame with the input sequence/timestamp used for that chunk

---

# Part 1 — Turn your current folder into a GitHub repo and push it (no placeholders)

Run these commands **in the folder containing your project** (the folder that has `package.json` and `server/`).

## 1) Install Git + GitHub CLI

```bash
sudo apt-get update
sudo apt-get install -y git gh
```

## 2) Login to GitHub (interactive, but no placeholders)

```bash
gh auth login
```

## 3) Set git identity automatically from your authenticated GitHub user

```bash
GITHUB_LOGIN="$(gh api user -q .login)"
GITHUB_ID="$(gh api user -q .id)"

git config --global user.name "$GITHUB_LOGIN"
git config --global user.email "${GITHUB_ID}+${GITHUB_LOGIN}@users.noreply.github.com"
```

## 4) Initialize git + commit

```bash
git init
git add -A
git commit -m "Initial commit"
```

## 5) Create the GitHub repo and push (repo name chosen from your app: lingbot-world-pilot)

```bash
gh repo create lingbot-world-pilot --public --source=. --remote=origin --push
```

At this point, your code is on GitHub under your logged-in account.

---

# Part 2 — GPU machine setup (clone app + download models to exact paths)

Everything below is run on the **separate Linux machine with the RTX 5090**.

## 1) Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs \
  python3.10 python3.10-venv python3-pip \
  build-essential ffmpeg
```

Enable git-lfs:

```bash
git lfs install
```

## 2) Create required directories

```bash
mkdir -p /home/aw/Documents/github/_homelab
mkdir -p /home/aw/Documents/models
```

## 3) Clone your app repo into the required folder

Login to GitHub on this machine:

```bash
gh auth login
```

Clone using your authenticated account (no placeholders):

```bash
GITHUB_LOGIN="$(gh api user -q .login)"
gh repo clone "${GITHUB_LOGIN}/lingbot-world-pilot" /home/aw/Documents/github/_homelab/lingbot-world-pilot
```

## 4) Download the NF4 model repo to the required exact path

This will pull ~30GB via git-lfs:

```bash
git clone https://huggingface.co/cahlen/lingbot-world-base-cam-nf4 /home/aw/Documents/models/lingbot
```

---

# Part 3 — Update backend dependencies + backend code (FULL FILES)

## 3.1 Replace `server/requirements_server.txt` with this exact content

On GPU machine:

```bash
cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
```

Overwrite the file:

```bash
cat > server/requirements_server.txt << 'EOF'
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
pydantic==2.10.6
pillow==11.1.0
numpy==1.26.4
requests==2.32.3
websocket-client==1.8.0
EOF
```

> Note: I pinned `numpy==1.26.4` deliberately because many ML stacks still break on numpy 2.x.

---

## 3.2 Replace `server/main.py` with this exact content (FULL CODE)

```bash
cat > server/main.py << 'EOF'
import asyncio
import io
import json
import os
import sys
import time
import uuid
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# ============================================================
# Paths / Imports
# ============================================================

APP_ROOT = Path(__file__).resolve().parents[1]

# Model repo must be a directory that contains generate_prequant.py + weights subfolders
MODEL_REPO = Path(os.getenv("LINGBOT_MODEL_REPO", "/home/aw/Documents/models/lingbot")).resolve()

if not MODEL_REPO.exists():
    raise RuntimeError(f"LINGBOT_MODEL_REPO not found: {MODEL_REPO}")

# Ensure model repo is importable (generate_prequant.py lives there)
if str(MODEL_REPO) not in sys.path:
    sys.path.insert(0, str(MODEL_REPO))

try:
    from generate_prequant import WanI2V_PreQuant  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import WanI2V_PreQuant from generate_prequant.py. "
        f"Check that {MODEL_REPO} contains generate_prequant.py. Error: {e}"
    )

# ============================================================
# Protocol models
# ============================================================

class InputStateModel(BaseModel):
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    space: bool = False  # idle toggle
    mouseX: float = 0.0  # delta since last send
    mouseY: float = 0.0  # delta since last send


class InputMsg(BaseModel):
    type: str = Field(pattern="^input$")
    seq: int
    client_ts_ms: int
    state: InputStateModel


class TelemetryMsg(BaseModel):
    type: str = "telemetry"
    server_ts_ms: int
    fps: float
    bufferMs: float
    latencyMs: float
    generationTimeMs: float
    lastInputSeq: int
    lastInputClientTsMs: int


class CreateSessionResp(BaseModel):
    session_id: str
    ws_url: str
    ws_path: str
    resolution: str
    quality: str


# ============================================================
# Config
# ============================================================

QUALITY_TO_STEPS = {
    "latency": 8,
    "balanced": 16,
    "quality": 28,
}

QUALITY_TO_GUIDE = {
    "latency": 4.0,
    "balanced": 5.0,
    "quality": 6.0,
}

RES_TO_HW = {
    "480p": (480, 832),
    "720p": (720, 1280),
}

TARGET_FPS = float(os.getenv("LINGBOT_TARGET_FPS", "16"))
CHUNK_FRAMES = max(5, int(os.getenv("LINGBOT_CHUNK_FRAMES", "9")))

LOW_WATER_FRAMES = int(os.getenv("LINGBOT_LOW_WATER_FRAMES", "18"))
HIGH_WATER_FRAMES = int(os.getenv("LINGBOT_HIGH_WATER_FRAMES", "60"))

JPEG_QUALITY_LAT = int(os.getenv("LINGBOT_JPEG_QUALITY_LAT", "85"))
JPEG_QUALITY_HQ = int(os.getenv("LINGBOT_JPEG_QUALITY_HQ", "92"))

MAX_SESSIONS = int(os.getenv("LINGBOT_MAX_SESSIONS", "1"))
STOP_ON_DISCONNECT = os.getenv("LINGBOT_STOP_ON_DISCONNECT", "1") == "1"
KEEP_MODELS_ON_GPU = os.getenv("LINGBOT_KEEP_MODELS_ON_GPU", "0") == "1"

CORS_ORIGINS = os.getenv("LINGBOT_CORS_ORIGINS", "*").split(",")

# ============================================================
# Session state
# ============================================================

@dataclass
class CameraState:
    x: float = 0.0
    y: float = 1.6
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass
class SessionState:
    session_id: str
    prompt: str
    resolution: str
    quality: str
    created_ms: int

    # rolling init image
    init_img: Image.Image

    # latest input
    latest_input: InputStateModel = field(default_factory=InputStateModel)
    latest_input_ts_ms: int = 0
    latest_input_seq: int = 0

    camera: CameraState = field(default_factory=CameraState)

    # streaming
    frame_id: int = 0
    chunk_id: int = 0
    frame_queue: "asyncio.Queue[Tuple[Dict[str, Any], bytes]]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=180)
    )
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    generator_task: Optional[asyncio.Task] = None

    # perf
    last_chunk_gen_ms: float = 0.0
    last_chunk_fps: float = 0.0


# ============================================================
# App
# ============================================================

app = FastAPI(title="LingBot-World Local Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE: Optional[WanI2V_PreQuant] = None
PIPELINE_LOCK = asyncio.Lock()

SESSIONS: Dict[str, SessionState] = {}


# ============================================================
# Utils
# ============================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def jpeg_encode_rgb(rgb_u8: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(rgb_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def pack_binary_frame(header: Dict[str, Any], jpeg_bytes: bytes) -> bytes:
    hb = json.dumps(header).encode("utf-8")
    header_len = len(hb).to_bytes(4, byteorder="little", signed=False)
    return header_len + hb + jpeg_bytes

def make_default_init_image(resolution: str) -> Image.Image:
    h, w = RES_TO_HW[resolution]
    return Image.new("RGB", (w, h), (127, 127, 127))

def intrinsics_for_base_480x832(num_frames: int) -> np.ndarray:
    base_h, base_w = 480, 832
    fx = base_w / 2.0
    fy = base_w / 2.0
    cx = base_w / 2.0
    cy = base_h / 2.0
    K = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (num_frames, 1))
    return K

def rot_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float32)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cp, -sp],
                   [0.0,  sp,  cp]], dtype=np.float32)

    return Ry @ Rx

def build_camera_poses(
    cam: CameraState,
    inp: InputStateModel,
    frames: int,
    fps: float,
    move_m_s: float = 1.2,
    mouse_sens: float = 0.002,
) -> Tuple[np.ndarray, CameraState]:
    """
    Produces camera-to-world transforms [F,4,4].

    If inp.space is True (idle toggle), we ignore motion + mouse.
    If no WASD pressed, we still generate frames (poses are constant).
    """
    dt = 1.0 / fps
    out = np.zeros((frames, 4, 4), dtype=np.float32)

    cam2 = CameraState(cam.x, cam.y, cam.z, cam.yaw, cam.pitch)

    if inp.space:
        # full idle: no mouse / no movement
        mouse_dx = 0.0
        mouse_dy = 0.0
        w = a = s = d = False
    else:
        mouse_dx = float(inp.mouseX)
        mouse_dy = float(inp.mouseY)
        w, a, s, d = inp.w, inp.a, inp.s, inp.d

    cam2.yaw += mouse_dx * mouse_sens
    cam2.pitch += mouse_dy * mouse_sens
    cam2.pitch = clamp(cam2.pitch, -1.2, 1.2)

    for i in range(frames):
        forward = 0.0
        right = 0.0
        if w: forward += 1.0
        if s: forward -= 1.0
        if d: right += 1.0
        if a: right -= 1.0

        mag = np.hypot(forward, right)
        if mag > 1e-6:
            forward /= mag
            right /= mag

        siny, cosy = np.sin(cam2.yaw), np.cos(cam2.yaw)

        cam2.x += (siny * forward + cosy * right) * move_m_s * dt
        cam2.z += (cosy * forward - siny * right) * move_m_s * dt

        R = rot_yaw_pitch(cam2.yaw, cam2.pitch)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array([cam2.x, cam2.y, cam2.z], dtype=np.float32)
        out[i] = T

    return out, cam2

async def ensure_pipeline_loaded() -> WanI2V_PreQuant:
    global PIPELINE
    async with PIPELINE_LOCK:
        if PIPELINE is None:
            PIPELINE = WanI2V_PreQuant(
                checkpoint_dir=str(MODEL_REPO),
                t5_cpu=True,
            )

            # Optional speed knob. May exceed VRAM on some setups; default is OFF.
            if KEEP_MODELS_ON_GPU:
                try:
                    PIPELINE.low_noise_model.to(PIPELINE.device)
                    PIPELINE.high_noise_model.to(PIPELINE.device)

                    def _no_offload(t, boundary):
                        return PIPELINE.high_noise_model if t.item() >= boundary else PIPELINE.low_noise_model

                    PIPELINE._prepare_model_for_timestep = _no_offload  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"[warn] KEEP_MODELS_ON_GPU failed, continuing with default offload: {e}")

        return PIPELINE


# ============================================================
# HTTP endpoints
# ============================================================

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "ts_ms": now_ms(), "sessions": len(SESSIONS), "model_repo": str(MODEL_REPO)}

@app.post("/api/session", response_model=CreateSessionResp)
async def create_session(
    request: Request,
    prompt: str = Form(...),
    resolution: str = Form("480p"),
    quality: str = Form("balanced"),
    initImage: Optional[UploadFile] = File(None),
) -> CreateSessionResp:
    if resolution not in RES_TO_HW:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {list(RES_TO_HW.keys())}")
    if quality not in QUALITY_TO_STEPS:
        raise HTTPException(status_code=400, detail=f"quality must be one of {list(QUALITY_TO_STEPS.keys())}")

    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(status_code=409, detail="Session limit reached (single-GPU config). Stop the existing session first.")

    if initImage is not None:
        data = await initImage.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        h, w = RES_TO_HW[resolution]
        img = img.resize((w, h), Image.BICUBIC)
    else:
        # "No image provided": bootstrap with a neutral image (the model is i2v).
        img = make_default_init_image(resolution)

    sid = str(uuid.uuid4())
    st = SessionState(
        session_id=sid,
        prompt=prompt,
        resolution=resolution,
        quality=quality,
        created_ms=now_ms(),
        init_img=img,
    )
    SESSIONS[sid] = st

    ws_path = f"/ws/session/{sid}"
    base = str(request.base_url)  # e.g. http://HOST:8000/
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    ws_url = f"{ws_base}{ws_path}"

    return CreateSessionResp(
        session_id=sid,
        ws_url=ws_url,
        ws_path=ws_path,
        resolution=resolution,
        quality=quality,
    )

@app.delete("/api/session/{session_id}")
async def stop_session(session_id: str) -> Dict[str, Any]:
    st = SESSIONS.get(session_id)
    if not st:
        return {"ok": True, "stopped": False}

    st.stop_event.set()
    if st.generator_task:
        st.generator_task.cancel()

    # cleanup workdir
    workdir = APP_ROOT / "server" / ".work" / st.session_id
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)

    del SESSIONS[session_id]
    return {"ok": True, "stopped": True}


# ============================================================
# Generator loop
# ============================================================

async def generator_loop(st: SessionState) -> None:
    pipeline = await ensure_pipeline_loaded()

    sampling_steps = QUALITY_TO_STEPS[st.quality]
    guide_scale = QUALITY_TO_GUIDE[st.quality]

    h, w = RES_TO_HW[st.resolution]
    max_area = h * w

    workdir = APP_ROOT / "server" / ".work" / st.session_id
    workdir.mkdir(parents=True, exist_ok=True)

    while not st.stop_event.is_set():
        # Backpressure: if buffer is large, yield CPU
        if st.frame_queue.qsize() >= HIGH_WATER_FRAMES:
            await asyncio.sleep(0.01)
            continue

        # Snapshot input for the whole chunk (lowest-latency approach would be smaller chunks)
        inp = st.latest_input
        inp_seq = st.latest_input_seq
        inp_ts = st.latest_input_ts_ms

        poses, new_cam = build_camera_poses(
            st.camera,
            inp,
            frames=CHUNK_FRAMES,
            fps=TARGET_FPS,
        )
        st.camera = new_cam

        intr = intrinsics_for_base_480x832(num_frames=CHUNK_FRAMES)

        np.save(str(workdir / "poses.npy"), poses)
        np.save(str(workdir / "intrinsics.npy"), intr)

        this_chunk_id = st.chunk_id
        st.chunk_id += 1

        # Run GPU generation off the event loop
        t0 = time.time()
        video = await asyncio.to_thread(
            pipeline.generate,
            st.prompt,
            st.init_img,
            str(workdir),   # action_path expects poses.npy + intrinsics.npy
            max_area,
            CHUNK_FRAMES,
            sampling_steps,
            guide_scale,
            -1,             # seed random
        )
        gen_ms = (time.time() - t0) * 1000.0

        # Convert torch tensor -> numpy uint8 frames
        v = video.detach().float().cpu()
        v = ((v + 1.0) * 0.5 * 255.0).clamp(0, 255).byte()
        v = v.permute(1, 2, 3, 0).numpy()  # [F,H,W,C]

        # Rolling init = last frame
        st.init_img = Image.fromarray(v[-1], mode="RGB")

        st.last_chunk_gen_ms = float(gen_ms)
        st.last_chunk_fps = float(CHUNK_FRAMES) / max(1e-6, (gen_ms / 1000.0))

        jpeg_q = JPEG_QUALITY_HQ if st.quality == "quality" else JPEG_QUALITY_LAT

        for i in range(v.shape[0]):
            # Drop old frames if queue is full (latency > completeness)
            if st.frame_queue.full():
                try:
                    _ = st.frame_queue.get_nowait()
                except Exception:
                    pass

            jpeg = jpeg_encode_rgb(v[i], quality=jpeg_q)
            header = {
                "type": "frame",
                "session_id": st.session_id,
                "frame_id": st.frame_id,
                "chunk_id": this_chunk_id,
                "chunk_frame_idx": i,
                "w": w,
                "h": h,
                "format": "jpeg",
                "server_ts_ms": now_ms(),
                "input_seq": inp_seq,
                "input_client_ts_ms": inp_ts,
            }
            st.frame_id += 1
            await st.frame_queue.put((header, jpeg))

        # If buffer is low, generate again immediately; else yield briefly
        if st.frame_queue.qsize() < LOW_WATER_FRAMES:
            continue
        await asyncio.sleep(0.001)


# ============================================================
# WebSocket endpoint
# ============================================================

@app.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str) -> None:
    await ws.accept()

    st = SESSIONS.get(session_id)
    if not st:
        await ws.close(code=1008)
        return

    if st.generator_task is None or st.generator_task.done():
        st.stop_event.clear()
        st.generator_task = asyncio.create_task(generator_loop(st))

    last_telemetry_ms = 0
    telemetry_interval_ms = 250

    async def recv_inputs() -> None:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                if data.get("type") == "input":
                    inp = InputMsg(**data)
                    st.latest_input = inp.state
                    st.latest_input_ts_ms = int(inp.client_ts_ms)
                    st.latest_input_seq = int(inp.seq)
            except Exception:
                continue

    recv_task = asyncio.create_task(recv_inputs())

    try:
        while True:
            # Send frame if available
            try:
                header, jpeg = await asyncio.wait_for(st.frame_queue.get(), timeout=0.05)
                await ws.send_bytes(pack_binary_frame(header, jpeg))
            except asyncio.TimeoutError:
                pass

            # Telemetry
            tms = now_ms()
            if tms - last_telemetry_ms >= telemetry_interval_ms:
                last_telemetry_ms = tms
                buffer_ms = (st.frame_queue.qsize() / TARGET_FPS) * 1000.0

                # Approx "input age" (for now). Real input->display latency is computed client-side.
                latency_ms = float(tms - st.latest_input_ts_ms) if st.latest_input_ts_ms else 0.0

                tel = TelemetryMsg(
                    server_ts_ms=tms,
                    fps=float(st.last_chunk_fps),
                    bufferMs=float(buffer_ms),
                    latencyMs=float(latency_ms),
                    generationTimeMs=float(st.last_chunk_gen_ms),
                    lastInputSeq=int(st.latest_input_seq),
                    lastInputClientTsMs=int(st.latest_input_ts_ms),
                )
                await ws.send_text(tel.model_dump_json())

    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
        if STOP_ON_DISCONNECT:
            st.stop_event.set()
            if st.generator_task:
                st.generator_task.cancel()

            workdir = APP_ROOT / "server" / ".work" / st.session_id
            if workdir.exists():
                shutil.rmtree(workdir, ignore_errors=True)

            SESSIONS.pop(session_id, None)
EOF
```

---

# Part 4 — Create Python venv + install dependencies (models + server)

## 1) Create venv inside your app repo

```bash
cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

## 2) Install model repo requirements first

```bash
pip install -r /home/aw/Documents/models/lingbot/requirements.txt
```

## 3) Install server requirements

```bash
pip install -r /home/aw/Documents/github/_homelab/lingbot-world-pilot/server/requirements_server.txt
```

---

# Part 5 — Run the backend on the RTX 5090 (using your GPU UUID)

## 1) Export required environment variables

Run exactly:

```bash
export CUDA_VISIBLE_DEVICES=GPU-11481043-00bd-5b3f-02e7-3138b3f915be
export LINGBOT_MODEL_REPO=/home/aw/Documents/models/lingbot
export LINGBOT_MAX_SESSIONS=1
export LINGBOT_TARGET_FPS=16
export LINGBOT_CHUNK_FRAMES=9
export LINGBOT_LOW_WATER_FRAMES=18
export LINGBOT_HIGH_WATER_FRAMES=60
export LINGBOT_KEEP_MODELS_ON_GPU=0
export LINGBOT_STOP_ON_DISCONNECT=1
export LINGBOT_CORS_ORIGINS=*
```

> If you later confirm VRAM headroom, you can switch `LINGBOT_KEEP_MODELS_ON_GPU=1` (faster, riskier for VRAM).

## 2) Start the server

Make sure venv is active:

```bash
cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
source .venv/bin/activate
```

Run:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

# Part 6 — Smoke test (guarantees server works before touching JS)

## 1) Health check

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

## 2) Create a test init image at /tmp/init.jpg (so no placeholders)

```bash
python3 - << 'PY'
from PIL import Image
img = Image.new("RGB", (832, 480), (120, 120, 120))
img.save("/tmp/init.jpg", "JPEG", quality=95)
print("/tmp/init.jpg written")
PY
```

## 3) Create a session via curl (returns session_id + ws_url)

```bash
curl -s \
  -F 'prompt=A futuristic cyberpunk city street at night, neon lights, rain on wet pavement, low angle view.' \
  -F 'resolution=480p' \
  -F 'quality=balanced' \
  -F 'initImage=@/tmp/init.jpg' \
  http://127.0.0.1:8000/api/session | python3 -m json.tool
```

## 4) Add a test client file that connects to WS and saves a few frames

Create `server/test_client.py`:

```bash
cat > server/test_client.py << 'EOF'
import json
import struct
import time
import requests
from websocket import create_connection

BASE = "http://127.0.0.1:8000"

def make_session():
    files = {
        "initImage": open("/tmp/init.jpg", "rb"),
    }
    data = {
        "prompt": "A futuristic cyberpunk city street at night, neon lights, rain on wet pavement, low angle view.",
        "resolution": "480p",
        "quality": "balanced",
    }
    r = requests.post(f"{BASE}/api/session", data=data, files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_frame_packet(b: bytes):
    header_len = struct.unpack("<I", b[:4])[0]
    header = json.loads(b[4:4+header_len].decode("utf-8"))
    jpeg = b[4+header_len:]
    return header, jpeg

def main():
    sess = make_session()
    ws_url = sess["ws_url"]
    print("WS:", ws_url)

    ws = create_connection(ws_url, timeout=120)

    # send a few inputs while receiving frames
    seq = 0
    saved = 0
    start = time.time()

    while saved < 10 and (time.time() - start) < 300:
        # send input (walk forward)
        msg = {
            "type": "input",
            "seq": seq,
            "client_ts_ms": int(time.time() * 1000),
            "state": {"w": True, "a": False, "s": False, "d": False, "space": False, "mouseX": 0.0, "mouseY": 0.0}
        }
        ws.send(json.dumps(msg))
        seq += 1

        # receive either telemetry (text) or frame (binary)
        frame = ws.recv()
        if isinstance(frame, str):
            try:
                t = json.loads(frame)
                if t.get("type") == "telemetry":
                    print("telemetry:", {k: t[k] for k in ["fps","bufferMs","generationTimeMs","lastInputSeq"]})
            except Exception:
                pass
            continue

        header, jpeg = parse_frame_packet(frame)
        out = f"/tmp/lingbot_frame_{header['frame_id']:06d}.jpg"
        with open(out, "wb") as f:
            f.write(jpeg)
        saved += 1
        print("saved", out, "hdr:", {k: header[k] for k in ["frame_id","chunk_id","chunk_frame_idx","input_seq"]})

    ws.close()
    # stop session
    requests.delete(f"{BASE}/api/session/{sess['session_id']}", timeout=30)
    print("done")

if __name__ == "__main__":
    main()
EOF
```

Run it:

```bash
source .venv/bin/activate
python3 server/test_client.py
```

After it finishes, verify files exist:

```bash
ls -lh /tmp/lingbot_frame_*.jpg | head
```

If those JPEGs exist and look plausible, your backend is working end-to-end.

---

## What happens next

Once you confirm the smoke test works, we’ll replace your mock loop in `WorldViewport.tsx` with:

* `POST /api/session` (multipart)
* WebSocket connect to `ws_url`
* send `InputMsg` at ~30Hz
* parse binary frame packets and draw JPEGs into the `<canvas>`

When you’re ready, tell me **the hostname/IP** you’ll use to access the GPU machine from the browser (LAN IP is fine), and I’ll give you the updated React code wired to this backend.
