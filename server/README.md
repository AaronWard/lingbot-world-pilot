


## Installation steps (NF4 repo + backend deps)

```sh 
cd Documents/github/_homelab/lingbot-world-pilot/
conda activate lingbot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


cd /home/aw/Documents/models/lingbot
pip install -r requirements.txt


cd Documents/github/_homelab/lingbot-world-pilot/
pip install -r server/requirements_server.txt
```

## Run Server

```sh
# export CUDA_VISIBLE_DEVICES=GPU-11481043-00bd-5b3f-02e7-3138b3f915be
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=GPU-11481043-00bd-5b3f-02e7-3138b3f915be   # the 5090 UUID
export CUDA_VISIBLE_DEVICES=1
export LINGBOT_MODEL_REPO=/home/aw/Documents/models/lingbot
export LINGBOT_MAX_SESSIONS=1
export LINGBOT_TARGET_FPS=4
export LINGBOT_CHUNK_FRAMES=1
export LINGBOT_LOW_WATER_FRAMES=2
export LINGBOT_HIGH_WATER_FRAMES=6
export LINGBOT_KEEP_MODELS_ON_GPU=0
export LINGBOT_STOP_ON_DISCONNECT=1
export LINGBOT_CORS_ORIGINS=*


conda activate lingbot 

cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```


## Run Web Application

```sh
cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
npm install
npm run dev
```


---


## Sanity Test


```sh
conda activate lingbot 

# Do this immediately after creating the session:
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

curl -s -X DELETE http://127.0.0.1:8000/api/session/62aa95a1-b5a2-44da-9e02-4264b8910015 | python3 -m json.tool


bash -lc '
set -u

echo "== Backend health ==";
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool || true;
echo;

echo "== Active listener on 8000 ==";
ss -ltnp 2>/dev/null | grep -E ":(8000)\b" || echo "Nothing listening on :8000";
echo;

echo "== GPU usage snapshot (read-only) ==";
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits;
echo;

echo "== Model folder sizes (read-only) ==";
du -sh /home/aw/Documents/models/lingbot 2>/dev/null || true;
ls -lh /home/aw/Documents/models/lingbot/models_t5_umt5-xxl-enc-bf16.pth \
      /home/aw/Documents/models/lingbot/Wan2.1_VAE.pth \
      /home/aw/Documents/models/lingbot/high_noise_model_bnb_nf4/model.safetensors \
      /home/aw/Documents/models/lingbot/low_noise_model_bnb_nf4/model.safetensors 2>/dev/null || true;
'


cd /home/aw/Documents/github/_homelab/lingbot-world-pilot
python3 server/test_client.py
```