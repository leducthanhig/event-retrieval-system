# Development
## Pre-requisites
- Python 3.10+
- Node.js
- npm
- FFmpeg
- Docker
- GNU Make (Optional, for Makefile commands)
## Setup
### Backend
- Install Python dependencies:
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
pip install 'numpy<2' matplotlib pandas tensorflow opencv-python \
    ffmpeg-python pillow ftfy regex tqdm elasticsearch 'fastapi[standard]' \
    open_clip_torch

GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
python backend/download-weights.py

conda install -c pytorch faiss-cpu
```
- Set up Elasticsearch:
```bash
curl -fsSL https://elastic.co/start-local | sh -s -- --esonly
```
### Frontend
- Install Node.js dependencies:
```bash
cd frontend
npm install
```
## Run the application
### Backend
- Start Elasticsearch (if not already running):
```bash
elastic-start-local/start.sh
```
- Export Elasticsearch local API key:
```bash
source elastic-start-local/.env
export ES_LOCAL_API_KEY
```
- Start the FastAPI server:
```bash
fastapi dev backend/src/app.py
```
### Frontend
```bash
npm run dev --prefix frontend
```
### Miscellaneous
- To configure the application, edit the `backend/src/config.py` file.
- To extract video frames:
```bash
python backend/src/examples/extract.py
```
- To index feature vectors and object features (requires `ES_LOCAL_API_KEY` environment variable):
```bash
python backend/src/examples/index.py
```
- To perform a search, play with the search parameters in `backend/src/examples/search.py` and run (requires `ES_LOCAL_API_KEY` environment variable):
```bash
python backend/src/examples/search.py
```
