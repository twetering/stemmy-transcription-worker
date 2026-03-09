#!/usr/bin/env python3
"""
RunPod Serverless: faster-whisper large-v3-turbo → AssemblyAI-compatibel JSON
met word-level timestamps.

Input:
  {"audio_url": "https://...", "item_id": "uuid", "beam_size": 5, "language": "nl"}
  {"audio_base64": "...",      "item_id": "uuid"}

Output (AssemblyAI-compatibel):
  {
    "item_id": "...",
    "text": "...",
    "words": [{"text": "...", "start": 0, "end": 120, "speaker": "A", "confidence": 0.97}],
    "utterances": [...],
    "language_code": "nl",
    "duration_seconds": 3612.4,
    "processing_seconds": 142.1
  }

Model: Systran/faster-whisper-large-v3-turbo
  - ~25× realtime op L4/RTX4090
  - word_timestamps=True altijd aan
  - Geen batch_size (dat is WhisperX) — wij gebruiken beam_size
"""

import base64
import os
import tempfile
import time
from pathlib import Path

import runpod

# ── Model eenmalig laden bij container-start (buiten handler) ────────────────
# Dit voorkomt dat elke job opnieuw GPU-init en model-load doet.
# RunPod hergebruikt de container tussen jobs → model blijft in GPU-geheugen.

print("[startup] Loading faster-whisper large-v3-turbo...", flush=True)
_load_start = time.time()

from faster_whisper import WhisperModel

_MODEL_PATH = "/app/models/faster-whisper-large-v3-turbo"
_MODEL = WhisperModel(
    _MODEL_PATH if os.path.isdir(_MODEL_PATH) else "large-v3-turbo",
    device="cuda",
    compute_type="float16",
)

print(f"[startup] Model ready in {time.time() - _load_start:.1f}s", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _download_audio(url: str, out_path: Path) -> bool:
    import httpx
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[error] Download failed: {e}", flush=True)
        return False


def _segments_to_words(segments, speaker: str = "A") -> list:
    """Faster-whisper segments → AssemblyAI words list."""
    words = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for w in seg_words:
                text = w.word.strip()
                if not text:
                    continue
                words.append({
                    "text": text,
                    "start": int(w.start * 1000),
                    "end": int(w.end * 1000),
                    "speaker": speaker,
                    "confidence": round(float(getattr(w, "probability", 0.9) or 0.9), 4),
                })
        elif seg.text.strip():
            # Fallback: segment-level als word_timestamps niet beschikbaar
            words.append({
                "text": seg.text.strip(),
                "start": int(seg.start * 1000),
                "end": int(seg.end * 1000),
                "speaker": speaker,
                "confidence": 0.9,
            })
    return words


def _words_to_utterances(words: list) -> list:
    """Groepeer words in utterances per speaker (voor diarization-compatibiliteit)."""
    if not words:
        return []
    utterances, cur_words, cur_speaker = [], [], words[0].get("speaker", "A")
    for w in words:
        sp = w.get("speaker", "A")
        if sp != cur_speaker and cur_words:
            utterances.append(_make_utterance(cur_words, cur_speaker))
            cur_words, cur_speaker = [], sp
        cur_words.append(w)
    if cur_words:
        utterances.append(_make_utterance(cur_words, cur_speaker))
    return utterances


def _make_utterance(words: list, speaker: str) -> dict:
    text = " ".join(w["text"] for w in words)
    conf = sum(w.get("confidence", 0.9) for w in words) / len(words)
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": speaker,
        "confidence": round(conf, 4),
        "words": words,
    }


def _transcribe(audio_path: Path, beam_size: int = 5, language: str = None) -> dict:
    t0 = time.time()

    segments, info = _MODEL.transcribe(
        str(audio_path),
        beam_size=beam_size,
        word_timestamps=True,
        language=language,  # None = auto-detect
    )
    segments = list(segments)  # generator evalueren

    processing_s = round(time.time() - t0, 1)
    duration_s = round(info.duration, 1) if info.duration else 0.0

    words = _segments_to_words(segments)
    full_text = " ".join(w["text"] for w in words)

    print(
        f"[transcribe] {duration_s}s audio → {len(words)} words "
        f"in {processing_s}s ({duration_s / max(processing_s, 1):.1f}× realtime), "
        f"lang={info.language}",
        flush=True,
    )

    return {
        "text": full_text,
        "words": words,
        "utterances": _words_to_utterances(words),
        "language_code": info.language or "nl",
        "duration_seconds": duration_s,
        "processing_seconds": processing_s,
    }


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job):
    inp = job.get("input", {})
    audio_url = inp.get("audio_url")
    audio_b64 = inp.get("audio_base64")
    item_id = inp.get("item_id", "unknown")
    beam_size = int(inp.get("beam_size", 5))
    language = inp.get("language") or None  # None = auto-detect

    if not audio_url and not audio_b64:
        return {"error": "Need audio_url or audio_base64", "item_id": item_id}

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.mp3"

        if audio_url:
            print(f"[handler] Downloading {audio_url[:80]}...", flush=True)
            if not _download_audio(audio_url, audio_path):
                return {"error": f"Download failed: {audio_url}", "item_id": item_id}
        else:
            try:
                audio_path.write_bytes(base64.b64decode(audio_b64))
            except Exception as e:
                return {"error": f"base64 decode failed: {e}", "item_id": item_id}

        try:
            result = _transcribe(audio_path, beam_size=beam_size, language=language)
            result["item_id"] = item_id
            return result
        except Exception as e:
            import traceback
            print(f"[error] Transcription failed: {traceback.format_exc()}", flush=True)
            return {"error": str(e), "item_id": item_id}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
