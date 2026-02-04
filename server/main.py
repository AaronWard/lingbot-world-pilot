import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io

# --- Ensure repo root on path so we can import generate_prequant.py ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NF4 pipeline (in this repo)
from generate_prequant import WanI2V_PreQuant  # type: ignore


# =========================
# Protocol models
# =========================

class InputStateModel(BaseModel):
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    space: bool = False  # idle toggle
    mouseX: float = 0.0  # delta
    mouseY: float = 0.0  # delta


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


class CreateSessionResp(BaseModel):
    session_id: str
    ws_url: str
    resolution: str
    quality: str


# =========================
# Session config + state
# =========================

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
    "480p": (480, 832),   # matches NF4 defaults
    "720p": (720, 1280),
}

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

    # rolling init image (PIL)
    init_img: Image.Image

    # input + camera
    latest_input: InputStateModel = field(default_factory=InputStateModel)
    latest_input_ts_ms: int = 0
    camera: CameraState = field(default_factory=CameraState)

    # streaming
    frame_id: int = 0
    frame_queue: "asyncio.Queue[Tuple[Dict[str, Any], bytes]]" = field(default_factory=lambda: asyncio.Queue(maxsize=180))
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    generator_task: Optional[asyncio.Task] = None

    # performance
    last_chunk_gen_ms: float = 0.0
    last_chunk_fps: float = 0.0


# =========================
# FastAPI app
# =========================

app = FastAPI(title="LingBot-World Local Backend", version="0.1.0")

# Dev-friendly CORS (local-first). Tighten later if you expose this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global single pipeline to keep VRAM stable
PIPELINE: Optional[WanI2V_PreQuant] = None
PIPELINE_LOCK = asyncio.Lock()

# One active session by default (practical on single GPU).
SESSIONS: Dict[str, SessionState] = {}
MAX_SESSIONS = int(os.getenv("LINGBOT_MAX_SESSIONS", "1"))


# =========================
# Utilities
# =========================

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def jpeg_encode_rgb(rgb_u8: np.ndarray, quality: int = 85) -> bytes:
    """
    rgb_u8: HxWx3 uint8
    """
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
    # Neutral gray gives the model "something" without strong bias.
    return Image.new("RGB", (w, h), (127, 127, 127))

def intrinsics_for_base_480x832(num_frames: int) -> np.ndarray:
    """
    NF4 code transforms intrinsics from base (480x832) to target output.
    So we provide intrinsics in base coordinates (fx, fy, cx, cy).
    """
    base_h, base_w = 480, 832
    # Approx ~90deg HFOV
    fx = base_w / 2.0
    fy = base_w / 2.0
    cx = base_w / 2.0
    cy = base_h / 2.0
    K = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (num_frames, 1))
    return K

def rot_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Yaw around Y, pitch around X (FPS-ish)
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
    fps: float = 16.0,
    move_m_s: float = 1.2,
    mouse_sens: float = 0.002,
) -> Tuple[np.ndarray, CameraState]:
    """
    Produce poses.npy as camera-to-world transforms [F,4,4] and updated camera state.
    We apply constant input for the whole chunk (chunk latency tradeoff).
    """
    dt = 1.0 / fps
    out = np.zeros((frames, 4, 4), dtype=np.float32)

    cam2 = CameraState(cam.x, cam.y, cam.z, cam.yaw, cam.pitch)

    # Apply mouse deltas once per chunk (client typically accumulates)
    cam2.yaw += float(inp.mouseX) * mouse_sens
    cam2.pitch += float(inp.mouseY) * mouse_sens
    cam2.pitch = clamp(cam2.pitch, -1.2, 1.2)

    # Movement per frame
    for i in range(frames):
        forward = 0.0
        right = 0.0
        if inp.w: forward += 1.0
        if inp.s: forward -= 1.0
        if inp.d: right += 1.0
        if inp.a: right -= 1.0

        # normalize diagonal
        mag = np.hypot(forward, right)
        if mag > 1e-6:
            forward /= mag
            right /= mag

        # Heading from yaw only
        siny, cosy = np.sin(cam2.yaw), np.cos(cam2.yaw)
        # +z forward in our state
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
            # Create once; heavy load.
            PIPELINE = WanI2V_PreQuant(
                checkpoint_dir=str(REPO_ROOT),
                t5_cpu=True,
            )

            # Optional speed knob: keep both experts on GPU to avoid swapping.
            keep_on_gpu = os.getenv("LINGBOT_KEEP_MODELS_ON_GPU", "0") == "1"
            if keep_on_gpu:
                try:
                    PIPELINE.low_noise_model.to(PIPELINE.device)
                    PIPELINE.high_noise_model.to(PIPELINE.device)

                    def _no_offload(t, boundary):
                        return PIPELINE.high_noise_model if t.item() >= boundary else PIPELINE.low_noise_model

                    PIPELINE._prepare_model_for_timestep = _no_offload  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"[warn] KEEP_MODELS_ON_GPU failed: {e}")

        return PIPELINE


# =========================
# HTTP endpoints
# =========================

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "ts_ms": now_ms(), "sessions": len(SESSIONS)}

@app.post("/api/session", response_model=CreateSessionResp)
async def create_session(
    prompt: str = Form(...),
    resolution: str = Form("480p"),
    quality: str = Form("balanced"),
    initImage: Optional[UploadFile] = File(None),
) -> CreateSessionResp:
    if resolution not in RES_TO_HW:
        raise ValueError(f"resolution must be one of {list(RES_TO_HW.keys())}")
    if quality not in QUALITY_TO_STEPS:
        raise ValueError(f"quality must be one of {list(QUALITY_TO_STEPS.keys())}")

    if len(SESSIONS) >= MAX_SESSIONS:
        raise RuntimeError("GPU backend currently configured for a single active session. Stop the existing one first.")

    if initImage is not None:
        data = await initImage.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Resize to exact target dims expected by the model path
        h, w = RES_TO_HW[resolution]
        img = img.resize((w, h), Image.BICUBIC)
    else:
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

    return CreateSessionResp(
        session_id=sid,
        ws_url=f"ws://localhost:8000/ws/session/{sid}",
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
    del SESSIONS[session_id]
    return {"ok": True, "stopped": True}


# =========================
# Generator loop
# =========================

async def generator_loop(st: SessionState) -> None:
    """
    Continuously generate small chunks and push JPEG frames into st.frame_queue.
    Uses rolling init image = last frame from previous chunk.
    """
    pipeline = await ensure_pipeline_loaded()

    target_fps = 16.0
    chunk_frames = int(os.getenv("LINGBOT_CHUNK_FRAMES", "9"))  # keep small for control latency
    # enforce >= 5
    chunk_frames = max(5, chunk_frames)

    # buffering thresholds in frames
    low_water = int(os.getenv("LINGBOT_LOW_WATER_FRAMES", "18"))    # ~1.1s at 16fps
    high_water = int(os.getenv("LINGBOT_HIGH_WATER_FRAMES", "60"))  # ~3.7s

    sampling_steps = QUALITY_TO_STEPS[st.quality]
    guide_scale = QUALITY_TO_GUIDE[st.quality]

    h, w = RES_TO_HW[st.resolution]
    max_area = h * w

    # workspace for action_path
    workdir = REPO_ROOT / "server" / ".work" / st.session_id
    workdir.mkdir(parents=True, exist_ok=True)

    while not st.stop_event.is_set():
        # backpressure: wait if buffer is full enough
        if st.frame_queue.qsize() >= high_water:
            await asyncio.sleep(0.01)
            continue

        # Snapshot current input for the whole chunk
        inp = st.latest_input
        poses, new_cam = build_camera_poses(
            st.camera,
            inp,
            frames=chunk_frames,
            fps=target_fps,
        )
        st.camera = new_cam

        intr = intrinsics_for_base_480x832(num_frames=chunk_frames)

        np.save(str(workdir / "poses.npy"), poses)
        np.save(str(workdir / "intrinsics.npy"), intr)

        # Generate chunk
        t0 = time.time()
        # Run generation off the event loop thread (GPU-bound)
        video = await asyncio.to_thread(
            pipeline.generate,
            st.prompt,
            st.init_img,
            str(workdir),             # action_path
            max_area,
            chunk_frames,             # frame_num
            sampling_steps,
            guide_scale,
            -1,                       # seed (random)
        )
        gen_ms = (time.time() - t0) * 1000.0

        # video is torch.Tensor shaped [C, F, H, W], range [-1, 1] :contentReference[oaicite:3]{index=3}
        # Convert to uint8 frames
        # We avoid importing torch at module top in case the environment is finicky; tensor ops are simple.
        v = video.detach().float().cpu()
        v = ((v + 1.0) * 0.5 * 255.0).clamp(0, 255).byte()
        # [C, F, H, W] -> [F, H, W, C]
        v = v.permute(1, 2, 3, 0).numpy()

        # Update rolling init image to last frame
        last = v[-1]
        st.init_img = Image.fromarray(last, mode="RGB")

        # Compute telemetry stats
        chunk_fps = float(chunk_frames) / max(1e-6, (gen_ms / 1000.0))
        st.last_chunk_gen_ms = gen_ms
        st.last_chunk_fps = chunk_fps

        # Enqueue frames
        for i in range(v.shape[0]):
            # Drop frames if queue is jammed (prefer low latency)
            if st.frame_queue.full():
                try:
                    _ = st.frame_queue.get_nowait()
                except Exception:
                    pass

            rgb = v[i]
            jpeg = jpeg_encode_rgb(rgb, quality=85 if st.quality != "quality" else 92)

            header = {
                "type": "frame",
                "session_id": st.session_id,
                "frame_id": st.frame_id,
                "w": w,
                "h": h,
                "format": "jpeg",
                "server_ts_ms": now_ms(),
            }
            st.frame_id += 1
            await st.frame_queue.put((header, jpeg))

        # If buffer is low, immediately generate again; else yield a bit.
        if st.frame_queue.qsize() < low_water:
            continue
        await asyncio.sleep(0.001)


# =========================
# WebSocket endpoint
# =========================

@app.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str) -> None:
    await ws.accept()

    st = SESSIONS.get(session_id)
    if not st:
        await ws.close(code=1008)
        return

    # Start generator if not running
    if st.generator_task is None or st.generator_task.done():
        st.stop_event.clear()
        st.generator_task = asyncio.create_task(generator_loop(st))

    # Duplex operation:
    # - read input messages
    # - send frames + periodic telemetry
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
                    st.latest_input_ts_ms = inp.client_ts_ms
            except Exception:
                # ignore malformed input
                continue

    recv_task = asyncio.create_task(recv_inputs())

    try:
        while True:
            # Send next frame if available
            try:
                header, jpeg = await asyncio.wait_for(st.frame_queue.get(), timeout=0.05)
                await ws.send_bytes(pack_binary_frame(header, jpeg))
            except asyncio.TimeoutError:
                pass

            # Send telemetry periodically
            tms = now_ms()
            if tms - last_telemetry_ms >= telemetry_interval_ms:
                last_telemetry_ms = tms
                buffer_ms = (st.frame_queue.qsize() / 16.0) * 1000.0

                # "latency" here is an approximation: time since last client input timestamp
                # until now; later we can compute true input->frame latency by tagging frames.
                latency_ms = float(tms - st.latest_input_ts_ms) if st.latest_input_ts_ms else 0.0

                tel = TelemetryMsg(
                    server_ts_ms=tms,
                    fps=float(st.last_chunk_fps),
                    bufferMs=float(buffer_ms),
                    latencyMs=float(latency_ms),
                    generationTimeMs=float(st.last_chunk_gen_ms),
                )
                await ws.send_text(tel.model_dump_json())

    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
        # Optional: stop session when client disconnects
        auto_stop = os.getenv("LINGBOT_STOP_ON_DISCONNECT", "1") == "1"
        if auto_stop:
            st.stop_event.set()
            if st.generator_task:
                st.generator_task.cancel()
            SESSIONS.pop(session_id, None)
