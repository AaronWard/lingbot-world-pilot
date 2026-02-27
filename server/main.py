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
                t5_cpu=False,
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