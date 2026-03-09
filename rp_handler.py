#!/usr/bin/env python3
"""
RunPod Serverless handler: WhisperX transcription → AssemblyAI format.

Input:  {"audio_url": "...", "item_id": "..."} or {"audio_base64": "...", "item_id": "..."}
Output: {"text", "words", "utterances", "language_code", "item_id"}
"""

import base64
import tempfile
from pathlib import Path


def _to_assemblyai_words(segments, default_speaker="A"):
    """Convert WhisperX segments to AssemblyAI words (ms, text, speaker, confidence)."""
    words = []
    for seg in segments:
        w = seg.get("word") or seg.get("text", "")
        if not w:
            continue
        start_ms = int(float(seg.get("start", 0) or 0) * 1000)
        end_ms = int(float(seg.get("end", 0) or 0) * 1000)
        conf = seg.get("score") or seg.get("confidence", 0.9)
        words.append({
            "text": w.strip(),
            "start": start_ms,
            "end": end_ms,
            "speaker": seg.get("speaker", default_speaker),
            "confidence": float(conf) if conf is not None else 0.9,
        })
    return words


def _words_to_utterances(words):
    """Group consecutive words with same speaker into utterances."""
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
    import whisperx
    import gc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    model = whisperx.load_model(
        "large-v3-turbo",
        device,
        compute_type=compute_type,
        asr_options={"suppress_numerals": True},
    )
    audio = whisperx.load_audio(str(audio_path))
    if audio is None or len(audio) == 0:
        return {"text": "", "words": [], "utterances": [], "language_code": "nl"}

    result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not result.get("segments"):
        return {"text": "", "words": [], "utterances": [], "language_code": "nl"}

    lang = result.get("language", "nl")
    align_model, metadata = whisperx.load_align_model(lang, device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    del align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    segments = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            word = w.get("word", "").strip()
            if not word:
                continue
            segments.append({
                "word": word,
                "start": w.get("start", seg.get("start", 0)),
                "end": w.get("end", seg.get("end", 0)),
                "score": w.get("score"),
                "speaker": "A",
            })

    words = _to_assemblyai_words(segments)
    full_text = " ".join(w["text"] for w in words)
    return {
        "text": full_text,
        "words": words,
        "utterances": _words_to_utterances(words),
        "language_code": lang,
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
