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
    header = json.loads(b[4:4 + header_len].decode("utf-8"))
    jpeg = b[4 + header_len:]
    return header, jpeg


def main():
    sess = make_session()
    ws_url = sess["ws_url"]
    print("WS:", ws_url)

    ws = create_connection(ws_url, timeout=120)

    seq = 0
    saved = 0
    start = time.time()

    while saved < 10 and (time.time() - start) < 300:
        msg = {
            "type": "input",
            "seq": seq,
            "client_ts_ms": int(time.time() * 1000),
            "state": {
                "w": True,
                "a": False,
                "s": False,
                "d": False,
                "space": False,
                "mouseX": 0.0,
                "mouseY": 0.0,
            },
        }
        ws.send(json.dumps(msg))
        seq += 1

        frame = ws.recv()
        if isinstance(frame, str):
            try:
                t = json.loads(frame)
                if t.get("type") == "telemetry":
                    print(
                        "telemetry:",
                        {k: t[k] for k in ["fps", "bufferMs", "generationTimeMs", "lastInputSeq"]},
                    )
            except Exception:
                pass
            continue

        header, jpeg = parse_frame_packet(frame)
        out = f"/tmp/lingbot_frame_{header['frame_id']:06d}.jpg"
        with open(out, "wb") as f:
            f.write(jpeg)
        saved += 1
        print(
            "saved",
            out,
            "hdr:",
            {k: header[k] for k in ["frame_id", "chunk_id", "chunk_frame_idx", "input_seq"]},
        )

    ws.close()
    requests.delete(f"{BASE}/api/session/{sess['session_id']}", timeout=30)
    print("done")


if __name__ == "__main__":
    main()