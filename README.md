
> [!IMPORTANT]  
> Currently a work in progress - I have a working WASD movement with [lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam) - but the framework is insufferably slow (~1 FPS). Gonna wait until **LingBot-World-Fast** is released. 


![splash](./splash.png)


## Run Locally
**Prerequisites:**  Node.js, python, CUDA

1. Install dependencies:
   `npm install`
2. Run the app:
   `npm run dev`


3. In a separate terminal, run the python websocket server
```bash
cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
pip install -r lingbot-world/requirements.txt fastapi uvicorn pillow
pip install flash-attn --no-build-isolation

export LW_REPO=$PWD/lingbot-world
export LW_CKPT_DIR=/mnt/data4tb/lingbot-world-base-cam
export LW_DEVICE_ID=0        # the 5090
export LW_T5_CPU=1           # 4060 (8GB) can't hold umt5-xxl; CPU is fine — T5 runs once/session then caches
export LW_LOCAL_ATTN=12      # ~3s window, ~15GB KV
export LW_SINK=1
export LW_CHUNK=3
export LW_SHIFT=3.0          # 480p
export LW_QUANT=nf4          # required to fit; wire your bnb swap into _maybe_quantize first
python server/main.py        # serves ws://0.0.0.0:8000/ws
```


![splash](./render.png)


---

### Quantize the model:

```sh
LW_REPO=$PWD/lingbot-world 
LW_CKPT_DIR=/mnt/data4tb/lingbot-world-base-cam 
LW_DEVICE_ID=0

python quantize_fast_nf4.py

export LW_FAST_SUBFOLDER=lingbot_world_fast_nf4
export LW_PREQUANTIZED=1
```