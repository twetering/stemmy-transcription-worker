#!/usr/bin/env python3
"""
RunPod Serverless: faster-whisper transcription → AssemblyAI format.
Geen WhisperX, geen PyTorch – alleen CTranslate2.

Input:  {"audio_url": "...", "item_id": "..."} of {"audio_base64": "...", "item_id": "..."}
Output: {"text", "words", "utterances", "language_code", "item_id"}
"""

import base64
import tempfile
from pathlib import Path


def _to_assemblyai_words(segments, default_speaker="A"):
    """Convert faster-whisper segments naar AssemblyAI words."""
    words = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for w in seg_words:
                if not w.word.strip():
                    continue
                words.append({
                    "text": w.word.strip(),
                    "start": int(w.start * 1000),
                    "end": int(w.end * 1000),
                    "speaker": default_speaker,
                    "confidence": getattr(w, "probability", 0.9) or 0.9,
                })
        elif seg.text.strip():
            # Fallback: segment-level als geen word timestamps
            words.append({
                "text": seg.text.strip(),
                "start": int(seg.start * 1000),
                "end": int(seg.end * 1000),
                "speaker": default_speaker,
                "confidence": 0.9,
            })
    return words


def _words_to_utterances(words):
    """Group words into utterances (zelfde speaker)."""
    if not words:
        return []
    utterances = []
    cur_speaker = words[0].get("speaker", "A")
    cur_words = []
    cur_text = []

    for w in words:
        sp = w.get("speaker", "A")
        if sp != cur_speaker and cur_words:
            utterances.append({
                "text": " ".join(cur_text),
                "start": cur_words[0]["start"],
                "end": cur_words[-1]["end"],
                "speaker": cur_speaker,
                "confidence": sum(x.get("confidence", 0.9) for x in cur_words) / len(cur_words),
                "words": cur_words.copy(),
            })
            cur_speaker = sp
            cur_words = []
            cur_text = []
        cur_words.append(w)
        cur_text.append(w["text"])

    if cur_words:
        utterances.append({
            "text": " ".join(cur_text),
            "start": cur_words[0]["start"],
            "end": cur_words[-1]["end"],
            "speaker": cur_speaker,
            "confidence": sum(x.get("confidence", 0.9) for x in cur_words) / len(cur_words),
            "words": cur_words,
        })
    return utterances


def _download_audio(url, out_path):
    import httpx
    try:
        r = httpx.get(url, follow_redirects=True, timeout=120)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


def _transcribe(audio_path, batch_size=16):
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3-turbo",
        device="cuda",
        compute_type="float16",
    )
    segments, info = model.transcribe(
        str(audio_path),
        batch_size=batch_size,
        word_timestamps=True,
        suppress_numerals=True,
    )
    segments = list(segments)

    if not segments:
        return {"text": "", "words": [], "utterances": [], "language_code": info.language or "nl"}

    words = _to_assemblyai_words(segments)
    full_text = " ".join(w["text"] for w in words)
    return {
        "text": full_text,
        "words": words,
        "utterances": _words_to_utterances(words),
        "language_code": info.language or "nl",
    }


def handler(job):
    inp = job.get("input", {})
    audio_url = inp.get("audio_url")
    audio_b64 = inp.get("audio_base64")
    item_id = inp.get("item_id", "unknown")
    batch_size = int(inp.get("batch_size", 16))

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.mp3"

        if audio_url:
            if not _download_audio(audio_url, audio_path):
                return {"error": f"Failed to download {audio_url}", "item_id": item_id}
        elif audio_b64:
            try:
                audio_path.write_bytes(base64.b64decode(audio_b64))
            except Exception as e:
                return {"error": str(e), "item_id": item_id}
        else:
            return {"error": "Need audio_url or audio_base64", "item_id": item_id}

        try:
            out = _transcribe(audio_path, batch_size=batch_size)
            out["item_id"] = item_id
            return out
        except Exception as e:
            return {"error": str(e), "item_id": item_id}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
