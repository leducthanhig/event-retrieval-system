# Development
## Pre-requisites
- Python 3.10+
- Micromamba
- Node.js
- npm
- FFmpeg
- GNU Make (Optional, for Makefile commands)
## Setup
### Backend
- Install Python dependencies:
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
pip install 'numpy<2' matplotlib pandas tensorflow \
    ffmpeg-python pillow ftfy regex tqdm 'fastapi[standard]' \
    open_clip_torch transformers

GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
python backend/download-weights.py

micromamba install -c pytorch faiss-cpu
```
### Frontend
- Install Node.js dependencies:
```bash
cd frontend
npm install
```
## Run the application
### Backend
- Activate the environment
```bash
eval "$(/c/micromamba/micromamba shell hook --shell bash)"
micromamba activate ner-backend
```
- Run
```bash
fastapi dev backend/src/app.py
```
### Frontend
```bash
npm run dev --prefix frontend
```
## Miscellaneous
- To configure the application, edit the `backend/src/config.py` file.
- To extract video frames:
```bash
python backend/src/extract.py
```
- To index feature vectors:
```bash
python backend/src/index.py
```
