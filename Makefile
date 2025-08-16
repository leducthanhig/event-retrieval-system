setup:
# 	install packages
	pip install torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cu118
	pip install matplotlib tensorflow ffmpeg-python pillow \
		ftfy regex tqdm "fastapi[standard]" open_clip_torch \
		google-genai python-dotenv accelerate elasticsearch \
		git+https://github.com/huggingface/transformers.git
	micromamba install -c pytorch faiss-cpu
	micromamba install ffmpeg

# 	install transnetv2 and download weights files manually due to git lfs issues
	GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
	python backend/download-weights.py

start-backend:
	fastapi dev backend/src/app.py

start-frontend:
	npm run dev --prefix frontend

install-es:
	curl -fsSL https://elastic.co/start-local | sh -s -- --esonly

start-es:
	elastic-start-local/start.sh

stop-es:
	elastic-start-local/stop.sh
