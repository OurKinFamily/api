FROM python:3.12-slim

WORKDIR /app

# ffmpeg: video posters/duration for uploads. exiftool: GPS write-back
# (gallery location editor) + mpp's own extraction backend.
# NOTE: the phone-upload pipeline (app/services/upload.py) also needs `mpp` (a
# Node CLI, not a pip dep) on PATH. Provision it in the deploy stack — bind-mount
# the host binary + Node runtime, or install the private package here — before
# enabling uploads on the deployed container. Local dev already has mpp on PATH.
# libjpeg-turbo-progs is jpegtran, which is what makes a crop lossless. Without
# it the crop service falls back to re-encoding, quietly — every crop on the
# deployed archive was costing a generation of quality, which is the one thing
# cropping a family photograph must not do.
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io ffmpeg libimage-exiftool-perl libjpeg-turbo-progs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY jobs.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
