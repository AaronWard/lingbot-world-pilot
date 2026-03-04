import asyncio
import io
import json
import os
import sys
import time
import uuid
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

import torch
torch.set_float32_matmul_precision('high')


# ============================================================
# Paths / Imports
# ============================================================

LINGBOT_T5_CPU = os.getenv("LINGBOT_T5_CPU", "1") == "1"
LINGBOT_PRELOAD_ON_STARTUP = os.getenv("LINGBOT_PRELOAD_ON_STARTUP", "1") == "1"
LINGBOT_FORCE_RESOLUTION = os.getenv("LINGBOT_FORCE_RESOLUTION", "").strip()

APP_ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO = Path(os.getenv("LINGBOT_MODEL_REPO", "/home/aw/Documents/models/lingbot")).resolve()
if not MODEL_REPO.exists():
    raise RuntimeError(f"LINGBOT_MODEL_REPO not found: {MODEL_REPO}")

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
    space: bool = False
    mouseX: float = 0.0
    mouseY: float = 0.0

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

QUALITY_TO_STEPS = {"latency": 8, "balanced": 12, "quality": 16}
QUALITY_TO_GUIDE = {"latency": 2.0, "balanced": 5.0, "quality": 6.0}

RES_TO_HW = {"480p": (480, 832), "720p": (720, 1280)}

TARGET_FPS = float(os.getenv("LINGBOT_TARGET_FPS", "16"))
CHUNK_FRAMES = max(1, int(os.getenv("LINGBOT_CHUNK_FRAMES", "5")))

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
    init_img: Image.Image

    latest_input: InputStateModel = field(default_factory=InputStateModel)
    latest_input_ts_ms: int = 0
    latest_input_seq: int = 0

    camera: CameraState = field(default_factory=CameraState)

    frame_id: int = 0
    chunk_id: int = 0
    frame_queue: "asyncio.Queue[Tuple[Dict[str, Any], bytes]]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=180)
    )
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    generator_task: Optional[asyncio.Task] = None

    last_chunk_gen_ms: float = 0.0
    last_chunk_fps: float = 0.0

# ============================================================
# App
# ============================================================

app = FastAPI(title="LingBot-World Local Backend", version="0.2.0")

logger = logging.getLogger("lingbot.server")
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,  # safer; you don't need cookies here
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE: Optional[WanI2V_PreQuant] = None
PIPELINE_LOCK = asyncio.Lock()
SESSIONS: Dict[str, SessionState] = {}

# ============================================================
# Startup
# ============================================================

def log_runtime_environment() -> None:
    logger.info("MODEL_REPO=%s", MODEL_REPO)
    logger.info("MAX_SESSIONS=%s", MAX_SESSIONS)
    logger.info("TARGET_FPS=%s", TARGET_FPS)
    logger.info("CHUNK_FRAMES=%s", CHUNK_FRAMES)
    logger.info("KEEP_MODELS_ON_GPU=%s", KEEP_MODELS_ON_GPU)
    logger.info("LINGBOT_T5_CPU=%s", LINGBOT_T5_CPU)
    logger.info("LINGBOT_FORCE_RESOLUTION=%s", LINGBOT_FORCE_RESOLUTION or "(none)")
    try:
        import torch
        logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("torch.cuda.device_count=%s", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                logger.info("cuda:%d name=%s", i, torch.cuda.get_device_name(i))
            logger.info("torch.cuda.current_device=%s", torch.cuda.current_device())
    except Exception as e:
        logger.warning("Failed to inspect torch CUDA environment: %s", e)

@app.on_event("startup")
async def on_startup() -> None:
    log_runtime_environment()
    if LINGBOT_PRELOAD_ON_STARTUP:
        logger.info("Preloading pipeline on startup...")
        await ensure_pipeline_loaded()

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
    return np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (num_frames, 1))

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
    move_m_s: float = 2.0,
    mouse_sens: float = 0.004,
) -> Tuple[np.ndarray, CameraState]:
    dt = 1.0 / fps
    out = np.zeros((frames, 4, 4), dtype=np.float32)
    cam2 = CameraState(cam.x, cam.y, cam.z, cam.yaw, cam.pitch)

    if inp.space:
        mouse_dx = mouse_dy = 0.0
        w = a = s = d = False
    else:
        mouse_dx = float(inp.mouseX)
        mouse_dy = float(inp.mouseY)
        w, a, s, d = inp.w, inp.a, inp.s, inp.d

    cam2.yaw += mouse_dx * mouse_sens
    cam2.pitch = clamp(cam2.pitch + mouse_dy * mouse_sens, -1.2, 1.2)

    for i in range(frames):
        forward = (1.0 if w else 0.0) + (-1.0 if s else 0.0)
        right = (1.0 if d else 0.0) + (-1.0 if a else 0.0)

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
            logger.info("Loading WanI2V_PreQuant...")

            import torch as _torch
            # With PCI_BUS_ID: cuda:0 = 4060 (8GB), cuda:1 = 5090 (32GB)
            # T5-XXL is ~10GB in bf16 — too big for the 4060, so keep on CPU
            # DiTs + VAE go on the 5090
            dit_device_id = 1 if _torch.cuda.device_count() >= 2 else 0
            t5_device_str = "cpu"
            vae_device_str = "cuda:0" if _torch.cuda.device_count() >= 2 else None

            logger.info("T5 on CPU, DiTs on cuda:%d", dit_device_id)

            PIPELINE = WanI2V_PreQuant(
                checkpoint_dir=str(MODEL_REPO),
                device_id=dit_device_id,
                t5_cpu=True,
                t5_device_str=t5_device_str,
                vae_device_str=vae_device_str,
            )
            logger.info("WanI2V_PreQuant loaded successfully")

            # Preload both DiT models onto GPU immediately
            logger.info("Preloading both DiT models onto GPU...")
            PIPELINE._lazy_load_high()
            PIPELINE._lazy_load_low()
            PIPELINE.high_noise_model.to(PIPELINE.device)
            PIPELINE.low_noise_model.to(PIPELINE.device)
            logger.info("Both DiT models on GPU, ready for inference")

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
    if LINGBOT_FORCE_RESOLUTION:
        resolution = LINGBOT_FORCE_RESOLUTION

    if resolution not in RES_TO_HW:
        raise HTTPException(status_code=400, detail=f"resolution must be one of {list(RES_TO_HW.keys())}")
    if quality not in QUALITY_TO_STEPS:
        raise HTTPException(status_code=400, detail=f"quality must be one of {list(QUALITY_TO_STEPS.keys())}")

    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(status_code=409, detail="Session limit reached. Stop the existing session first.")

    if initImage is not None:
        data = await initImage.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
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

    ws_path = f"/ws/session/{sid}"
    base = str(request.base_url)
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

    workdir = APP_ROOT / "server" / ".work" / st.session_id
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)

    del SESSIONS[session_id]
    return {"ok": True, "stopped": True}

# ============================================================
# Generator loop
# ============================================================

async def generator_loop(st: SessionState) -> None:
    try:
        pipeline = await ensure_pipeline_loaded()

        sampling_steps = QUALITY_TO_STEPS[st.quality]
        guide_scale = QUALITY_TO_GUIDE[st.quality]

        h, w = RES_TO_HW[st.resolution]
        max_area = h * w

        workdir = APP_ROOT / "server" / ".work" / st.session_id
        workdir.mkdir(parents=True, exist_ok=True)

        # Send the init image as frame 0 immediately so the user sees something
        import torchvision.transforms.functional as TF_server
        init_np = np.array(st.init_img)
        jpeg_q = JPEG_QUALITY_HQ if st.quality == "quality" else JPEG_QUALITY_LAT
        jpeg = jpeg_encode_rgb(init_np, quality=jpeg_q)
        header = {
            "type": "frame",
            "session_id": st.session_id,
            "frame_id": st.frame_id,
            "chunk_id": -1,
            "chunk_frame_idx": 0,
            "w": w,
            "h": h,
            "format": "jpeg",
            "server_ts_ms": now_ms(),
            "input_seq": 0,
            "input_client_ts_ms": 0,
        }
        st.frame_id += 1
        await st.frame_queue.put((header, jpeg))

        # Keep a raw tensor for the conditioning image to avoid JPEG roundtrip
        # This is the key quality fix: no lossy compression between chunks
        raw_last_frame_tensor = None  # Will be set after first chunk

        while not st.stop_event.is_set():
            if st.frame_queue.qsize() >= HIGH_WATER_FRAMES:
                await asyncio.sleep(0.01)
                continue

            # Sample input as late as possible for responsiveness
            inp = st.latest_input
            inp_seq = st.latest_input_seq
            inp_ts = st.latest_input_ts_ms

            poses, new_cam = build_camera_poses(st.camera, inp, frames=CHUNK_FRAMES, fps=TARGET_FPS)
            st.camera = new_cam

            intr = intrinsics_for_base_480x832(num_frames=CHUNK_FRAMES)

            np.save(str(workdir / "poses.npy"), poses)
            np.save(str(workdir / "intrinsics.npy"), intr)

            this_chunk_id = st.chunk_id
            st.chunk_id += 1

            t0 = time.time()
            video = await asyncio.to_thread(
                pipeline.generate,
                input_prompt=st.prompt,
                img=st.init_img,
                raw_init_tensor=raw_last_frame_tensor,  # Pass raw tensor
                action_path=str(workdir),
                max_area=max_area,
                frame_num=CHUNK_FRAMES,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=-1,
            )
            gen_ms = (time.time() - t0) * 1000.0

            # Keep the raw last frame tensor for next chunk (NO JPEG roundtrip)
            # raw_last_frame_tensor = video[:, -1:, :, :].detach().clone()
            raw_last_frame_tensor = video[:, -1:, :, :].detach().to(PIPELINE.device).clone()


            v = video.detach().float().cpu()
            v = ((v + 1.0) * 0.5 * 255.0).clamp(0, 255).byte()
            v = v.permute(1, 2, 3, 0).numpy()

            # Still update init_img for PIL fallback, but raw_last_frame_tensor
            # is what actually gets used for conditioning
            st.init_img = Image.fromarray(v[-1], mode="RGB")

            st.last_chunk_gen_ms = float(gen_ms)
            st.last_chunk_fps = float(CHUNK_FRAMES) / max(1e-6, (gen_ms / 1000.0))

            for i in range(v.shape[0]):
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

            if st.frame_queue.qsize() < LOW_WATER_FRAMES:
                continue
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        logger.info("generator_loop cancelled for session %s", st.session_id)
        raise
    except Exception:
        logger.exception("generator_loop failed for session %s", st.session_id)
        st.stop_event.set()

# ============================================================
# WebSocket endpoint
# ============================================================
@app.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str) -> None:
    accepted = False
    closed = False

    try:
        await ws.accept()
        accepted = True
    except RuntimeError:
        return

    st = SESSIONS.get(session_id)
    if not st:
        if accepted and not closed:
            try:
                await ws.close(code=1008)
            except RuntimeError:
                pass
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
            if st.stop_event.is_set() and (st.generator_task is None or st.generator_task.done()) and st.frame_queue.empty():
                if not closed:
                    try:
                        await ws.close(code=1011, reason="generation stopped")
                    except RuntimeError:
                        pass
                    closed = True
                break

            try:
                header, jpeg = await asyncio.wait_for(st.frame_queue.get(), timeout=0.05)
                if not closed:
                    try:
                        await ws.send_bytes(pack_binary_frame(header, jpeg))
                    except RuntimeError:
                        closed = True
                        break
            except asyncio.TimeoutError:
                pass

            tms = now_ms()
            if tms - last_telemetry_ms >= telemetry_interval_ms:
                last_telemetry_ms = tms
                buffer_ms = (st.frame_queue.qsize() / TARGET_FPS) * 1000.0

                if st.latest_input_ts_ms:
                    latency_ms = max(0.0, float(tms - st.latest_input_ts_ms))
                else:
                    latency_ms = 0.0

                tel = TelemetryMsg(
                    server_ts_ms=tms,
                    fps=float(st.last_chunk_fps),
                    bufferMs=float(buffer_ms),
                    latencyMs=float(latency_ms),
                    generationTimeMs=float(st.last_chunk_gen_ms),
                    lastInputSeq=int(st.latest_input_seq),
                    lastInputClientTsMs=int(st.latest_input_ts_ms),
                )

                if not closed:
                    try:
                        await ws.send_text(tel.model_dump_json())
                    except RuntimeError:
                        closed = True
                        break

    except WebSocketDisconnect:
        closed = True
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