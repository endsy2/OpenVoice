"""Khmer voice cloning: a Khmer base TTS + OpenVoice V2 tone color converter.

Usage:
  .venv/bin/python khmer_clone.py "សួស្តី ខ្ញុំសប្បាយចិត្តណាស់" --ref resources/example_reference.mp3 --out outputs_v2/khmer.wav
  .venv/bin/python khmer_clone.py --speech my_khmer.m4a --ref resources/example_reference.mp3   # re-voice a real recording

--base picks the Khmer TTS that speaks the text (the converter then swaps in the --ref voice):
  gemini:Kore / gemini:Puck (any Gemini prebuilt voice name works)
      Google gemini-3.8-flash-tts, prompted to talk casually. Most human-sounding; needs GEMINI_API_KEY,
      text is sent to Google.
  edge:km-KH-SreymomNeural / edge:km-KH-PisethNeural
      Microsoft neural voices (female / male) via edge-tts. Most natural; text is sent to Microsoft.
  voxcpm:female / voxcpm:male
      OpenBMB VoxCPM2 (Apache-2.0), local and free; ~5 GB model, uses the Mac GPU (MPS). With --ref it
      clones the voice itself instead of going through the OpenVoice converter.
  facebook/mms-tts-khm (default), khmerttsopensource/khmer-tts, KrorngAI/mms-tts-khm-finetuned
      Local VITS models (CC-BY-NC, non-commercial).

The VITS models need transformers>=4.33 but MeloTTS pins 4.27.4, so on import this module relaunches
the running script via `uv run --with transformers==4.46.3 --with voxcpm` (the venv itself stays untouched).
"""
import argparse
import asyncio
import base64
import functools
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

try:
    from transformers import AutoTokenizer, VitsModel
except ImportError:
    if os.environ.get("KHMER_CLONE_RELAUNCHED"):
        raise
    os.environ["KHMER_CLONE_RELAUNCHED"] = "1"
    os.execvp("uv", ["uv", "run", "--no-project", "--python", sys.executable,
                     "--with", "transformers==4.46.3", "--with", "voxcpm", sys.argv[0], *sys.argv[1:]])

import numpy as np
import soundfile
import torch

from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Keys from .env (KEY=value lines); a variable already set in the shell wins.
if os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#") and v.strip():
            os.environ.setdefault(k.strip(), v.strip())

CKPT = "checkpoints_v2/converter"
SES_DIR = "checkpoints_v2/base_speakers/ses"
DEVICE = "cpu"
# name -> label, shown in the web UI; any other Hugging Face VITS id also works from the CLI.
BASES = {
    "gemini:Kore": "Google Gemini Kore (female, online, most natural)",
    "gemini:Puck": "Google Gemini Puck (male, online, most natural)",
    "voxcpm:female": "OpenBMB VoxCPM2 female (local, free)",
    "voxcpm:male": "OpenBMB VoxCPM2 male (local, free)",
    "edge:km-KH-SreymomNeural": "Microsoft Sreymom (female, online)",
    "edge:km-KH-PisethNeural": "Microsoft Piseth (male, online)",
    "khmerttsopensource/khmer-tts": "Community khmer-tts (local)",
    "KrorngAI/mms-tts-khm-finetuned": "KrorngAI MMS fine-tune (local)",
    "facebook/mms-tts-khm": "Meta MMS (local)",
}
# A few varied sentences so the base speaker embedding is stable (computed once per base, then cached).
CALIBRATION = (
    "សួស្តី ខ្ញុំឈ្មោះសុខា។ ថ្ងៃនេះអាកាសធាតុល្អណាស់។ ខ្ញុំចូលចិត្តអានសៀវភៅ និងដើរលេងនៅសួនច្បារ។ "
    "ប្រទេសកម្ពុជាមានប្រាសាទអង្គរវត្តដ៏ល្បីល្បាញ។ សូមអរគុណច្រើនសម្រាប់ការជួយរបស់អ្នក។ "
    "យើងនឹងជួបគ្នាម្តងទៀតនៅសប្តាហ៍ក្រោយ។ កុមារទាំងអស់កំពុងរៀននៅសាលា។"
)


# Sentence + its closing punctuation. Each sentence is synthesized on its own so we control the pauses.
SENT = re.compile(r"([^។?!៕\n]+)([។?!៕\n]*)")


def speak(text, path, synth, sr):
    """Human-ish delivery: each sentence gets slightly different pace/pitch (synth(sent, rng) decides how)
    and a breath-length pause that varies, instead of one flat newsreader take. Seeded by the text, so the
    same text always sounds the same."""
    text = re.sub(r"<[^>]*>", " ", text)  # Gemini-only tags like <breath>; other voices would read them aloud
    rng = random.Random(text)
    fade = int(0.01 * sr)
    chunks = []
    for sent, punct in SENT.findall(text):
        if not sent.strip():
            continue
        wav = synth(sent.strip(), rng)
        if wav is None or len(wav) < 2 * fade:
            continue
        wav = wav.astype(np.float32)
        wav[:fade] *= np.linspace(0, 1, fade)  # no clicks at the joins
        wav[-fade:] *= np.linspace(1, 0, fade)
        pause = rng.uniform(0.55, 0.85) if "\n" in punct else rng.uniform(0.25, 0.5)
        chunks += [wav, np.zeros(int(pause * sr), np.float32)]
    if not chunks:
        raise ValueError("No Khmer text to speak (these models only read Khmer script).")
    soundfile.write(path, np.concatenate(chunks[:-1]), sr)
    return path


# Picked by listening test: a short style beats a long one (long prompts drift). The bigger win is writing the
# text the way people talk: "...", commas, small fillers, and tags like <breath> or <short pause>.
GEMINI_STYLE = "casual and friendly"


def gemini_tts(text, path, voice, style=None):
    """One call for the whole text: Gemini plans the intonation across sentences itself, so no splitting."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY to use the Gemini voices.")
    body = {
        "model": "gemini-3.8-flash-tts",
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": GEMINI_STYLE if style is None else style}]}]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions", json.dumps(body).encode(),
        {"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                msg = json.loads(body)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                msg = body[:300]
            # Free tier: 3 requests/minute (worth waiting out) and 10/day (not).
            if e.code == 429 and "per day" not in msg and attempt < 3:
                time.sleep(21)
                continue
            if e.code == 429:
                msg += " Use a Microsoft voice meanwhile."
            raise ValueError(f"Gemini: {msg}") from None  # ValueError -> shown to the web user as a 400
    audio = [c for st in resp.get("steps", []) if st.get("type") == "model_output"
             for c in st.get("content", []) if c.get("type") == "audio"]
    if not audio:
        raise RuntimeError(f"Gemini TTS returned no audio: {str(resp)[:300]}")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio[-1]["data"]))  # 24 kHz mono WAV
    return path


# VoxCPM "voice design": a description in parentheses before the text picks the voice and delivery.
VOXCPM_VOICES = {
    # Picked by listening test (sample E): describe a chat with a friend, not just a voice.
    "voxcpm:female": ("A young Cambodian woman chatting with a close friend, natural relaxed pace, "
                      "a smile in her voice, everyday conversational Khmer"),
    "voxcpm:male": ("A young Cambodian man chatting with a close friend, natural relaxed pace, "
                    "a smile in his voice, everyday conversational Khmer"),
}
VOXCPM_STYLE = "casual and friendly"  # used when cloning: the reference decides the voice


@functools.lru_cache(maxsize=1)
def get_voxcpm():
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)  # device auto: mps on a Mac


def voxcpm_tts(text, path, voice=None, ref=None):
    """Design a voice from VOXCPM_VOICES, or clone `ref` directly (VoxCPM does its own voice cloning)."""
    text = re.sub(r"<[^>]*>", " ", text).strip()  # Gemini-only tags
    if not text:
        raise ValueError("No Khmer text to speak.")
    model = get_voxcpm()
    kw = {}
    if ref:
        import librosa
        wav, sr = librosa.load(ref, sr=None)  # any upload format -> wav VoxCPM can read
        ref_wav = path.rsplit(".", 1)[0] + "_ref.wav"
        soundfile.write(ref_wav, wav, sr)
        kw["reference_wav_path"] = ref_wav
    try:
        wav = model.generate(text=f"({VOXCPM_STYLE if ref else VOXCPM_VOICES[voice]}){text}",
                             cfg_value=1.5, inference_timesteps=20, **kw)  # looser = more lively
    finally:
        if ref:
            os.remove(kw["reference_wav_path"])  # don't keep copies of people's uploaded voices
    wav = wav * (0.9 / max(np.abs(wav).max(), 1e-6))  # VoxCPM comes out ~half as loud as the other voices
    soundfile.write(path, wav, model.tts_model.sample_rate)
    return path


@functools.lru_cache(maxsize=None)
def get_tts(base):
    """Return tts(text, path) -> written audio path. Models load once per base."""
    if base.startswith("gemini:"):
        return functools.partial(gemini_tts, voice=base.split(":", 1)[1])
    if base.startswith("voxcpm:"):
        return functools.partial(voxcpm_tts, voice=base)
    if base.startswith("edge:"):
        import edge_tts
        import librosa
        voice = base.split(":", 1)[1]

        def synth(sent, rng):
            # Microsoft's Khmer voices read a bit fast and flat: slow down a touch and wobble pitch per sentence.
            tmp = f"outputs_v2/_edge_{os.getpid()}.mp3"
            com = edge_tts.Communicate(sent, voice, rate=f"{rng.randint(-12, -2):+d}%", pitch=f"{rng.randint(-4, 3):+d}Hz")
            asyncio.run(com.save(tmp))
            wav, _ = librosa.load(tmp, sr=24000)
            os.remove(tmp)
            return librosa.effects.trim(wav, top_db=45)[0]  # we add our own pauses

        return lambda text, path: speak(text, path, synth, 24000)

    tok, model = AutoTokenizer.from_pretrained(base), VitsModel.from_pretrained(base).eval()
    model.noise_scale = 0.75  # default 0.667; a bit more life in the voice
    model.noise_scale_duration = 1.0  # default 0.8; less robotic, more uneven (human) syllable timing

    def synth(sent, rng):
        inputs = tok(sent, return_tensors="pt")
        if inputs["input_ids"].shape[-1] < 2:
            return None  # nothing Khmer left after normalization
        model.speaking_rate = rng.uniform(0.9, 1.0)
        with torch.no_grad():
            return model(**inputs).waveform[0].numpy()

    def tts(text, path):
        torch.manual_seed(0)  # VITS sampling is random; fixed seed = repeatable output
        return speak(text, path, synth, model.config.sampling_rate)
    return tts


@functools.lru_cache(maxsize=1)
def get_converter():
    tcc = ToneColorConverter(f"{CKPT}/config.json", device=DEVICE)
    tcc.load_ckpt(f"{CKPT}/checkpoint.pth")
    return tcc


def source_se(base):
    """Tone color of the base voice: computed once from CALIBRATION, then cached on disk."""
    slug = re.sub(r"[^A-Za-z0-9-]+", "_", base)
    se_path = f"{SES_DIR}/khm-{slug}.pth"
    if not os.path.exists(se_path):
        os.makedirs("outputs_v2", exist_ok=True)
        calib = get_tts(base)(CALIBRATION, f"outputs_v2/calibration_{slug}.wav")
        se, _ = se_extractor.get_se(calib, get_converter(), vad=True)
        torch.save(se.cpu(), se_path)
    return torch.load(se_path, map_location=DEVICE)


def clone(text, base, ref, out, tau=0.3, speech=None):
    """Speak `text` with `base` (or use `speech`, a real Khmer recording), then convert it to the voice in `ref`.
    Returns (out, base_audio_path)."""
    if base.startswith("voxcpm:") and not speech:  # clones by itself, better than TTS + converter
        return voxcpm_tts(text, out, ref=ref), None
    tcc = get_converter()
    tgt, _ = se_extractor.get_se(ref, tcc, vad=True)
    if speech:  # most human option: a person's own timing/intonation, only the voice gets swapped
        base_audio, src = speech, se_extractor.get_se(speech, tcc, vad=True)[0]
    else:
        base_audio, src = get_tts(base)(text, out.rsplit(".", 1)[0] + "_base.wav"), source_se(base)
    tcc.convert(audio_src_path=base_audio, src_se=src, tgt_se=tgt, output_path=out, tau=tau, message="@MyShell")
    # Pure digital silence between words is a TTS giveaway; lay a faint room-tone floor (~-60 dBFS) under it.
    wav, sr = soundfile.read(out)
    soundfile.write(out, wav + np.random.default_rng(0).normal(0, 1e-3, wav.shape), sr)
    return out, base_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="", help="Khmer text to speak (not needed with --speech)")
    ap.add_argument("--ref", required=True, help="recording of the voice to clone (any language)")
    ap.add_argument("--out", default="outputs_v2/khmer.wav")
    ap.add_argument("--base", default="facebook/mms-tts-khm", help="see module docstring")
    ap.add_argument("--tau", type=float, default=0.3, help="converter strength-ish knob (OpenVoice default 0.3)")
    ap.add_argument("--speech", help="your own Khmer recording to re-voice instead of TTS (sounds most human)")
    args = ap.parse_args()
    if not (args.text or args.speech):
        ap.error("give Khmer text or --speech")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out, base_audio = clone(args.text, args.base, args.ref, args.out, args.tau, args.speech)
    print("wrote", out, f"(base audio: {base_audio})" if base_audio else "")


if __name__ == "__main__":
    main()
