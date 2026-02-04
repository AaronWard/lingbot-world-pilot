


## Installation steps (NF4 repo + backend deps)

```sh 
git clone https://huggingface.co/cahlen/lingbot-world-base-cam-nf4
cd lingbot-world-base-cam-nf4


python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

pip install -r server/requirements_server.txt

```


## Run Server.


```sh
uvicorn server.main:app --host 0.0.0.0 --port 8000
```