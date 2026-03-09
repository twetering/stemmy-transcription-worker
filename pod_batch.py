#!/usr/bin/env python3
"""
pod_batch.py — Batch-transcriptie op een RunPod Pod (niet serverless).

Gebruik:
  pip install faster-whisper httpx feedparser
  python pod_batch.py --rss https://castopod.hku.nl/@HKUenAI/feed.xml
  python pod_batch.py --rss https://... --limit 5 --output ./results/
  python pod_batch.py --rss https://... --compare-dir ./assemblyai_jsons/  # benchmark

Output: JSON-bestanden per aflevering, AssemblyAI-compatibel formaat.
"""

import argparse
import json
import time
import tempfile
from pathlib import Path

import httpx
import feedparser
from faster_whisper import WhisperModel


def load_model(model_size="large-v3-turbo"):
    print(f"[load] Loading {model_size}...", flush=True)
    t0 = time.time()
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print(f"[load] Ready in {time.time()-t0:.1f}s", flush=True)
    return model


def download(url: str, path: Path):
    with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes(65536):
                f.write(chunk)


def transcribe(model, audio_path: Path, language=None, beam_size=5) -> dict:
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        word_timestamps=True,
        language=language,
    )
    segments = list(segments)
    proc = round(time.time() - t0, 1)
    dur = round(info.duration or 0, 1)

    words = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            if w.word.strip():
                words.append({
                    "text": w.word.strip(),
                    "start": int(w.start * 1000),
                    "end": int(w.end * 1000),
                    "speaker": "A",
                    "confidence": round(float(getattr(w, "probability", 0.9) or 0.9), 4),
                })

    print(f"  → {dur}s audio in {proc}s ({dur/max(proc,1):.1f}× realtime), {len(words)} words, lang={info.language}")
    return {
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "language_code": info.language or "nl",
        "duration_seconds": dur,
        "processing_seconds": proc,
    }


def benchmark(ours: dict, theirs_path: Path) -> dict:
    """Vergelijk word-count en spot-check tekst met AssemblyAI referentie."""
    with open(theirs_path) as f:
        ref = json.load(f)
    ref_words = ref.get("words", [])
    our_words = ours.get("words", [])

    # Word count vergelijking
    diff_pct = abs(len(our_words) - len(ref_words)) / max(len(ref_words), 1) * 100
    print(f"  [benchmark] words: ours={len(our_words)} ref={len(ref_words)} diff={diff_pct:.1f}%")

    return {
        "our_word_count": len(our_words),
        "ref_word_count": len(ref_words),
        "word_count_diff_pct": round(diff_pct, 1),
    }


def process_feed(rss_url: str, limit: int, output_dir: Path, compare_dir: Path = None,
                 language: str = None, beam_size: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()

    feed = feedparser.parse(rss_url)
    episodes = feed.entries[:limit] if limit else feed.entries
    print(f"[feed] {len(episodes)} afleveringen te verwerken")

    for i, ep in enumerate(episodes, 1):
        title = ep.get("title", f"episode_{i}")
        # Audio URL uit enclosure
        audio_url = None
        for enc in ep.get("enclosures", []):
            if "audio" in enc.get("type", ""):
                audio_url = enc.get("href") or enc.get("url")
                break
        if not audio_url:
            print(f"[{i}/{len(episodes)}] Geen audio URL voor: {title}")
            continue

        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:60].strip()
        out_path = output_dir / f"{i:03d}_{safe_title}.json"

        if out_path.exists():
            print(f"[{i}/{len(episodes)}] Skip (bestaat al): {safe_title}")
            continue

        print(f"[{i}/{len(episodes)}] {title[:60]}")
        print(f"  Downloading {audio_url[:70]}...")

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "audio.mp3"
            try:
                download(audio_url, audio_path)
            except Exception as e:
                print(f"  Download mislukt: {e}")
                continue

            result = transcribe(model, audio_path, language=language, beam_size=beam_size)
            result["item_id"] = ep.get("id", audio_url)
            result["title"] = title
            result["audio_url"] = audio_url

            # Benchmark tegen AssemblyAI referentie als beschikbaar
            if compare_dir:
                ref_candidates = list(compare_dir.glob(f"*{safe_title[:20]}*.json"))
                if ref_candidates:
                    bench = benchmark(result, ref_candidates[0])
                    result["benchmark"] = bench

            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"  Saved → {out_path.name}")

    print(f"\n✅ Klaar. Resultaten in {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Batch-transcriptie op RunPod Pod")
    ap.add_argument("--rss", required=True, help="RSS feed URL")
    ap.add_argument("--limit", type=int, default=0, help="Max afleveringen (0=alles)")
    ap.add_argument("--output", default="./results", help="Output map")
    ap.add_argument("--compare-dir", help="Map met AssemblyAI referentie-JSONs voor benchmark")
    ap.add_argument("--language", default=None, help="Taalcode (nl/en/auto) — default: auto-detect")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    process_feed(
        rss_url=args.rss,
        limit=args.limit,
        output_dir=Path(args.output),
        compare_dir=Path(args.compare_dir) if args.compare_dir else None,
        language=args.language if args.language != "auto" else None,
        beam_size=args.beam_size,
    )


if __name__ == "__main__":
    main()
