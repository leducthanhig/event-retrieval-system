# replace `micromamba` with your installed distribution
setup:
# 	install packages
	pip install "numpy<2" torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cu124
	pip install "numpy<2" matplotlib tensorflow ffmpeg-python \
		pillow ftfy regex tqdm "fastapi[standard]" open_clip_torch \
		transformers "huggingface_hub[hf_xet]" llama-cpp-python
	micromamba install -c pytorch faiss-cpu

# 	install transnetv2 and download weights files manually due to git lfs issues
	GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
	python backend/download-weights.py

# run this if ffmpeg binaries is not installed yet
install-ffmpeg:
	micromamba install ffmpeg

start-backend:
	fastapi dev backend/src/app.py

start-frontend:
	npm run dev --prefix frontend
