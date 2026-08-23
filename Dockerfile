FROM python:3.12-slim

WORKDIR /app

# ffmpeg: video posters/duration for uploads. exiftool: GPS write-back
# (gallery location editor) + mpp's own extraction backend.
# NOTE: the phone-upload pipeline (app/services/upload.py) also needs `mpp` (a
# Node CLI, not a pip dep) on PATH. Provision it in the deploy stack — bind-mount
# the host binary + Node runtime, or install the private package here — before
# enabling uploads on the deployed container. Local dev already has mpp on PATH.
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io ffmpeg libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY jobs.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
