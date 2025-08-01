# replace `micromamba` with your installed distribution
setup:
# 	install packages
	pip install torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cu118
	pip install 'numpy<2' matplotlib pandas tensorflow opencv-python \
		ffmpeg-python pillow ftfy regex tqdm elasticsearch 'fastapi[standard]' \
		open_clip_torch

# 	install transnetv2 and download weights files manually due to git lfs issues
	GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/soCzech/TransNetV2.git
	python backend/download-weights.py

# 	install cpu version of faiss on windows
	micromamba install -c pytorch faiss-cpu

#	unofficial version
#	pip install faiss-cpu

#	gpu versions are only available on linux
#	micromamba install -c pytorch -c nvidia faiss-gpu
#	micromamba install -c pytorch -c nvidia -c rapidsai -c conda-forge \
		libnvjitlink faiss-gpu-cuvs

# run this if ffmpeg binaries is not installed yet
install-ffmpeg:
	micromamba install ffmpeg

install-es:
	curl -fsSL https://elastic.co/start-local | sh -s -- --esonly

start-es:
	elastic-start-local/start.sh

stop-es:
	elastic-start-local/stop.sh

start-backend:
	fastapi dev backend/src/app.py

start-frontend:
	npm run dev --prefix frontend
