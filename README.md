# Development
## Pre-requisites
- Python 3.10+
- Micromamba
- Node.js
- npm
- Docker
- GNU Make (Optional, for Makefile commands)
- Gemini API key (create and get one [here](https://aistudio.google.com/apikey))
## Setup
### Backend
- Install Python dependencies:
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib tensorflow ffmpeg-python pillow \
    ftfy regex tqdm "fastapi[standard]" open_clip_torch \
    google-genai python-dotenv accelerate elasticsearch \
    faster-whisper ctranslate2==4.6.0 transformers
micromamba install ffmpeg
micromamba install -c pytorch faiss-cpu
micromamba install -c nvidia libcublas=12.6 cudnn=9

GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
python backend/download-weights.py
```
- Set up Elasticsearch:
```bash
curl -fsSL https://elastic.co/start-local | sh -s -- --esonly
```
- Create a `.env` file in the `backend` directory and add your secret keys:
```
GEMINI_API_KEY=your_api_key_here
ES_LOCAL_API_KEY=your_api_key_here
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
- Start the FastAPI server:
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
python backend/src/sample.py
```
- To extract feature vectors:
```bash
python backend/src/extract.py
```
- To index feature vectors:
```bash
python backend/src/index.py
```
