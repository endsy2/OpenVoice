"""Local web app for Khmer TTS + voice cloning.

  .venv/bin/python server.py      -> http://127.0.0.1:8000
"""
import os
import threading
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import khmer_clone as kc

OUT = "outputs_v2/web"
MAX_TEXT = 2000
MAX_UPLOAD = 30 * 1024 * 1024
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}

os.makedirs(OUT, exist_ok=True)
app = FastAPI()
# ponytail: one generation at a time (the models aren't thread-safe); add a job queue if several people share it.
lock = threading.Lock()


def validate(text, base):
    text = text.strip()
    if not text:
        raise HTTPException(400, "Enter some Khmer text.")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"Text is too long (max {MAX_TEXT} characters).")
    if base not in kc.BASES:
        raise HTTPException(400, "Unknown voice.")
    return text


def run(fn, *args):
    with lock:
        try:
            return fn(*args)
        except (ValueError, NotImplementedError) as e:  # empty Khmer text / no speech found in the reference
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/voices")
def voices():
    return [{"id": k, "label": v} for k, v in kc.BASES.items()]


@app.post("/api/tts")
def tts(text: str = Form(...), base: str = Form(...)):
    text = validate(text, base)
    return FileResponse(run(kc.get_tts(base), text, f"{OUT}/{uuid.uuid4().hex}_tts.wav"))


@app.post("/api/clone")
def clone(text: str = Form(...), base: str = Form(...), tau: float = Form(0.3), reference: UploadFile = File(...)):
    text = validate(text, base)
    if not 0.05 <= tau <= 1.0:
        raise HTTPException(400, "tau must be between 0.05 and 1.0.")
    ext = os.path.splitext(reference.filename or "")[1].lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Reference must be an audio file ({', '.join(sorted(AUDIO_EXT))}).")

    job = uuid.uuid4().hex
    ref = f"{OUT}/{job}_ref{ext}"
    size = 0
    with open(ref, "wb") as f:
        while chunk := reference.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                os.remove(ref)
                raise HTTPException(400, "Reference file is too large (max 30 MB).")
            f.write(chunk)
    try:
        out, _ = run(kc.clone, text, base, ref, f"{OUT}/{job}_clone.wav", tau)
    finally:
        os.remove(ref)
    return FileResponse(out)


@app.get("/")
def index():
    return FileResponse("web/index.html")


if __name__ == "__main__":
    kc.get_converter()  # load the 125 MB converter now, not on the first request
    uvicorn.run(app, host="127.0.0.1", port=8000)
