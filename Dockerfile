# CineScope web app + pipeline, CPU only.
#
#   docker build -t cinescope .
#   docker run --rm -p 8000:8000 -v "$PWD/trailers:/clips" -v "$PWD/data/weights:/weights" \
#              -v "$PWD/outputs:/out" cinescope
#   # then open http://localhost:8000
#
# Weights and footage are mounted, never baked in (see README: Footage and copyright).

# ---- front end: build the static files
FROM node:22-slim AS frontend
WORKDIR /src
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
RUN npm run build

# ---- back end
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2 \
    CINESCOPE_WEIGHTS=/weights CINESCOPE_CLIPS=/clips CINESCOPE_OUT=/out CINESCOPE_ONNX_DIR=/out/onnx \
    CINESCOPE_BACKEND=onnx
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
       fastapi uvicorn python-multipart onnx onnxruntime open_clip_torch faiss-cpu rapidocr-onnxruntime scikit-learn
COPY cinescope/ cinescope/
COPY app/__init__.py app/
COPY app/backend/ app/backend/
COPY --from=frontend /src/dist app/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
