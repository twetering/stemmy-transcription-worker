#!/usr/bin/env python3
"""
stemmy_batch.py — Complete stemmy transcription pipeline op RunPod Pod.

Pipeline per aflevering:
  1. Turso-check: skip als source_url al bestaat (dedupe)
  2. Download audio van RSS-enclosure
  3. Normalize: ffmpeg 192k CBR, 1ms trim, ID3v2.3, write_xing=0
     (identiek aan stemmy/surrounded format_service.py)
  4. Upload naar S3: audio/formats/{format_id}/{item_id}.mp3
  5. Transcribeer lokale normalized.mp3 met faster-whisper (GPU)
  6. Segments → AssemblyAI-compatibele utterances (met word-level timestamps)
  7. Schrijf format + item + fragments naar Turso

Vereisten:
  pip install faster-whisper httpx feedparser boto3
  apt-get install -y ffmpeg

Env vars:
  TURSO_URL          — libsql://... URL (zonder libsql://, gebruik https://)
  TURSO_TOKEN        — Turso auth token
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_REGION         (default: eu-north-1)
  S3_BUCKET_NAME     (default: voxpop)

Usage:
  python3 stemmy_batch.py --rss https://... --language nl
  python3 stemmy_batch.py --rss https://... --limit 5 --dry-run
  python3 stemmy_batch.py --rss https://... --format-id bestaand-uuid
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import feedparser
import httpx

# ── Turso HTTP client (Hrana v2 protocol) ────────────────────────────────────

def _turso_url() -> str:
    url = os.environ["TURSO_URL"]
    # libsql://foo.turso.io → https://foo.turso.io
    return url.replace("libsql://", "https://")

def _turso_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['TURSO_TOKEN']}",
        "Content-Type": "application/json",
    }

def _val(v):
    """Python waarde → Turso Hrana arg dict."""
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": str(v)}
    return {"type": "text", "value": str(v)}

def turso_execute(sql: str, params: list = None) -> list:
    """Voer één SQL statement uit. Retourneert rows als list[dict]."""
    stmt = {"sql": sql}
    if params:
        stmt["args"] = [_val(p) for p in params]

    resp = httpx.post(
        f"{_turso_url()}/v2/pipeline",
        headers=_turso_headers(),
        json={"requests": [{"type": "execute", "stmt": stmt}]},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    result = data["results"][0]
    if result.get("type") == "error":
        raise RuntimeError(f"Turso error: {result['error']}")

    rs = result.get("response", {}).get("result", {})
    cols = [c["name"] for c in rs.get("cols", [])]
    rows = []
    for row in rs.get("rows", []):
        rows.append({col: (cell["value"] if cell["type"] != "null" else None)
                     for col, cell in zip(cols, row)})
    return rows

def turso_insert(table: str, data: dict):
    cols = list(data.keys())
    placeholders = ", ".join("?" * len(cols))
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    turso_execute(sql, [data[c] for c in cols])


def turso_pipeline_batch(statements: list, chunk_size: int = 200):
    """
    Voer meerdere SQL-statements uit in batches van chunk_size per HTTP-call.
    Veel sneller dan losse calls: 465 inserts = 3 HTTP-calls i.p.v. 465.
    statements = [{"sql": "...", "args": [...]}, ...]
    """
    for i in range(0, len(statements), chunk_size):
        chunk = statements[i:i + chunk_size]
        requests = [
            {"type": "execute", "stmt": stmt}
            for stmt in chunk
        ]
        resp = httpx.post(
            f"{_turso_url()}/v2/pipeline",
            headers=_turso_headers(),
            json={"requests": requests},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        for j, result in enumerate(data.get("results", [])):
            if result.get("type") == "error":
                print(f"  [warn] Turso batch insert {i+j}: {result['error']}", flush=True)

# ── Audio normalisatie (identiek aan stemmy format_service) ──────────────────

NORMALIZE_BITRATE = "192k"

def normalize_audio(input_path: str, output_path: str):
    """
    ffmpeg: 1ms trim van eind, 192k CBR, ID3v2.3, write_xing=0.
    Identiek aan surrounded/services/format_service.py regel 691.
    """
    # Bepaal duur
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", input_path],
        capture_output=True, text=True
    )
    duration = None
    try:
        info = json.loads(probe.stdout)
        duration = float(info["format"]["duration"])
    except Exception:
        pass

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if duration:
        # Trim 1ms van eind
        end_time = duration - 0.001
        cmd += ["-t", str(end_time)]
    cmd += [
        "-b:a", NORMALIZE_BITRATE,
        "-write_xing", "0",
        "-write_id3v1", "1",
        "-id3v2_version", "3",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalisatie mislukt: {result.stderr.decode()[:300]}")

# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, s3_key: str) -> str:
    bucket = os.environ.get("S3_BUCKET_NAME", "voxpop")
    region = os.environ.get("AWS_REGION", "eu-north-1")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=region,
    )
    s3.upload_file(local_path, bucket, s3_key,
                   ExtraArgs={"ContentType": "audio/mpeg"})
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"

# ── Transcriptie (faster-whisper) ────────────────────────────────────────────

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        model_path = "/app/models/faster-whisper-large-v3-turbo"
        model_id = model_path if os.path.isdir(model_path) else "large-v3-turbo"
        print(f"[model] Loading {model_id}...", flush=True)
        t0 = time.time()
        _MODEL = WhisperModel(model_id, device="cuda", compute_type="float16")
        print(f"[model] Ready in {time.time()-t0:.1f}s", flush=True)
    return _MODEL

def transcribe(audio_path: str, language: str = None, beam_size: int = 5) -> dict:
    """
    Transcribeer audio. Retourneert AssemblyAI-compatibele dict:
    {text, utterances, words, language_code, duration_seconds, processing_seconds}

    Utterances = faster-whisper SEGMENTS (niet speaker-turns).
    Elk segment = 1 fragment in Turso met zijn words.
    """
    model = get_model()
    t0 = time.time()

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        word_timestamps=True,
        language=language,
    )
    segments = list(segments_gen)

    processing_s = round(time.time() - t0, 1)
    duration_s = round(info.duration or 0, 1)

    # Segments → utterances (AssemblyAI-compatible)
    utterances = []
    all_words = []
    for seg in segments:
        seg_words = []
        for w in (getattr(seg, "words", None) or []):
            if not w.word.strip():
                continue
            word = {
                "text": w.word.strip(),
                "start": int(w.start * 1000),
                "end": int(w.end * 1000),
                "speaker": "A",
                "confidence": round(float(getattr(w, "probability", 0.9) or 0.9), 4),
            }
            seg_words.append(word)
            all_words.append(word)

        if not seg_words and seg.text.strip():
            # Fallback: geen word-timestamps, gebruik segment-grenzen
            word = {
                "text": seg.text.strip(),
                "start": int(seg.start * 1000),
                "end": int(seg.end * 1000),
                "speaker": "A",
                "confidence": 0.9,
            }
            seg_words.append(word)
            all_words.append(word)

        if seg_words:
            avg_conf = round(sum(w["confidence"] for w in seg_words) / len(seg_words), 4)
            utterances.append({
                "text": " ".join(w["text"] for w in seg_words),
                "start": int(seg.start * 1000),
                "end": int(seg.end * 1000),
                "speaker": "A",
                "confidence": avg_conf,
                "words": seg_words,
            })

    full_text = " ".join(w["text"] for w in all_words)
    word_count = len(all_words)

    speed = round(duration_s / max(processing_s, 1), 1)
    print(f"  → {duration_s}s audio | {processing_s}s | {speed}× realtime | "
          f"{word_count} words | {len(utterances)} utterances | lang={info.language}",
          flush=True)

    return {
        "text": full_text,
        "utterances": utterances,
        "words": all_words,
        "language_code": info.language or "nl",
        "duration_seconds": duration_s,
        "processing_seconds": processing_s,
    }

# ── Turso import: format + item + fragments ───────────────────────────────────

def _title_to_identifier(title: str) -> str:
    """Genereer een shortname/identifier uit een format-titel.
    Consistent met bestaande identifiers: lowercase, underscores, max 40 tekens.
    Bijv: 'Ervaring voor Beginners' → 'ervaring_voor_beginners'
    """
    import re
    s = re.sub(r"[^\w\s]", "", title.lower(), flags=re.UNICODE)
    stopwords = {"de", "het", "een", "van", "en", "voor", "over", "bij", "op", "in",
                 "met", "aan", "om", "uit", "the", "a", "an", "of", "and", "for"}
    words = [w for w in s.split() if w not in stopwords] or s.split()
    return "_".join(words[:4])[:40]


def ensure_format(rss_url: str, title: str, description: str,
                  image_url: str, format_id: str = None) -> str:
    """
    Maak format aan als het nog niet bestaat. Retourneert format_id.

    Rules:
    - Altijd een UUID als format ID (nooit een slug)
    - Identifier (shortname) altijd invullen
    - Dedupe op source_url (niet op ID — die is random)
    - INSERT OR IGNORE: parallel pods zijn veilig
    """
    existing = turso_execute(
        "SELECT id FROM formats WHERE source_url = ?", [rss_url]
    )
    if existing:
        fid = existing[0]["id"]
        print(f"[format] Bestaand: {title} ({fid})", flush=True)
        return fid

    # Altijd UUID genereren — nooit een slug doorgeven als format_id
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    if format_id and uuid_pattern.match(format_id):
        fid = format_id
    else:
        if format_id:
            print(f"[format] Opgegeven format_id '{format_id}' is geen UUID → nieuwe UUID gegenereerd",
                  flush=True)
        fid = str(uuid.uuid4())

    identifier = _title_to_identifier(title)
    now = _now()
    turso_insert("formats", {
        "id": fid,
        "identifier": identifier,
        "title": title,
        "description": description or "",
        "source_url": rss_url,
        "source_type": "rss",
        "image_url": image_url or "",
        "created_at": now,
        "updated_at": now,
    })
    print(f"[format] Nieuw: {title} (id={fid}, identifier={identifier})", flush=True)
    return fid

def item_exists(format_id: str, source_url: str) -> tuple:
    """Retourneert (item_id, audio_url) of (None, None)."""
    rows = turso_execute(
        "SELECT id, audio_url FROM items WHERE format_id = ? AND source_url = ?",
        [format_id, source_url]
    )
    if rows:
        return rows[0]["id"], rows[0]["audio_url"]
    return None, None

def has_fragments(item_id: str) -> bool:
    """Controleer of er al fragmenten zijn voor dit item.
    Gebruik LIMIT 1 i.p.v. COUNT(*) — stopt zodra het eerste record gevonden is.
    Vereist idx_fragments_item_id voor snelle lookup.
    """
    rows = turso_execute(
        "SELECT id FROM fragments WHERE item_id = ? LIMIT 1", [item_id]
    )
    return len(rows) > 0

def save_item(item_id: str, format_id: str, ep: dict,
              source_url: str, audio_url: str):
    now = _now()
    turso_insert("items", {
        "id": item_id,
        "format_id": format_id,
        "title": ep["title"],
        "description": (ep.get("description") or "")[:2000],
        "audio_url": audio_url,
        "source_url": source_url,
        "published_at": ep.get("published", now),
        "duration_seconds": ep.get("duration", 0),
        "transcript_status": "pending",
        "created_at": now,
        "updated_at": now,
    })

def save_fragments(item_id: str, audio_url: str, transcript: dict):
    """
    Importeer utterances als fragments via Turso batch pipeline.
    Alle inserts in zo min mogelijk HTTP-calls (chunk van 200).
    GPU is klaar zodra transcriptie klaar is — Turso-inserts zijn non-GPU.
    """
    utterances = transcript.get("utterances", [])
    now = _now()
    meta = json.dumps({
        "source_info": {
            "type": "faster-whisper",
            "model": "large-v3-turbo",
            "imported_at": now,
            "total_fragments": len(utterances),
        }
    })

    cols = [
        "id", "item_id", "text", "start_time_seconds", "end_time_seconds",
        "start_time", "end_time", "duration", "speaker_label", "confidence",
        "words", "type", "source_type", "item_audio_url",
        "order_index", "created_at", "updated_at", "metadata",
    ]
    placeholders = ", ".join("?" * len(cols))
    insert_sql = f"INSERT OR IGNORE INTO fragments ({', '.join(cols)}) VALUES ({placeholders})"

    statements = []
    for idx, utt in enumerate(utterances):
        start_ms = utt.get("start", 0)
        end_ms = utt.get("end", 0)
        start_s = start_ms / 1000.0 if start_ms > 1000 else float(start_ms)
        end_s = end_ms / 1000.0 if end_ms > 1000 else float(end_ms)
        duration = round(end_s - start_s, 3)

        values = [
            str(uuid.uuid4()), item_id, utt.get("text", ""),
            str(start_s), str(end_s), start_ms, end_ms, str(duration),
            utt.get("speaker", "A"), str(utt.get("confidence", 0)),
            json.dumps(utt.get("words", [])),
            "transcript", "utterances", audio_url,
            str(idx), now, now, meta,
        ]
        statements.append({"sql": insert_sql, "args": [_val(v) for v in values]})

    t0 = time.time()
    turso_pipeline_batch(statements)
    turso_execute(
        "UPDATE items SET transcript_status = 'completed', updated_at = ? WHERE id = ?",
        [_now(), item_id]
    )
    turso_ms = round((time.time() - t0) * 1000)
    print(f"  💾 {len(utterances)} fragments → Turso in {turso_ms}ms "
          f"({len(statements)//200 + 1} HTTP-call(s))", flush=True)
    return len(utterances)

# ── RSS parse ─────────────────────────────────────────────────────────────────

def parse_rss(rss_url: str, limit: int = 0) -> tuple:
    """Retourneert (feed_info, episodes[])."""
    feed = feedparser.parse(rss_url)
    ch = feed.feed

    feed_info = {
        "title": ch.get("title", "Onbekend"),
        "description": ch.get("summary", ch.get("subtitle", "")),
        "image": (ch.get("image", {}).get("href") or
                  ch.get("itunes_image", {}).get("href") or ""),
    }

    episodes = []
    entries = feed.entries[:limit] if limit else feed.entries
    for ep in entries:
        audio_url = None
        for enc in ep.get("enclosures", []):
            if "audio" in enc.get("type", ""):
                audio_url = enc.get("href") or enc.get("url")
                break
        if not audio_url:
            continue

        # Duur parsen (HH:MM:SS of MM:SS of seconden)
        duration = 0
        dur_str = ep.get("itunes_duration", "")
        if dur_str:
            parts = str(dur_str).split(":")
            try:
                if len(parts) == 3:
                    duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    duration = int(parts[0]) * 60 + int(parts[1])
                else:
                    duration = int(parts[0])
            except Exception:
                pass

        episodes.append({
            "title": ep.get("title", ""),
            "description": ep.get("summary", ""),
            "audio_url": audio_url,
            "published": ep.get("published", ""),
            "duration": duration,
        })

    return feed_info, episodes

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def download(url: str, path: str):
    with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes(65536):
                f.write(chunk)

# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_pending_items(format_id: str, offset: int = 0, limit: int = 0) -> list:
    """
    Haal pending items op uit Turso voor een specifiek format.
    Retourneert items als list[dict] met id, title, source_url, audio_url.
    Offset/limit voor parallelle pods.
    """
    rows = turso_execute(
        "SELECT id, title, source_url, audio_url, duration_seconds "
        "FROM items WHERE format_id = ? AND transcript_status = 'pending' "
        "ORDER BY title",
        [format_id]
    )
    if offset:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    return rows


def run_rss_mode(args, lang: str):
    """Standaard RSS-mode: parse feed, verwerk episodes."""
    print(f"[rss] Fetching {args.rss}...", flush=True)
    fetch_limit = (args.offset + args.limit) if args.limit else 0
    feed_info, all_episodes = parse_rss(args.rss, fetch_limit)
    episodes = all_episodes[args.offset:] if args.offset else all_episodes
    total_rss = len(all_episodes)
    print(f"[rss] {feed_info['title']} — {total_rss} totaal, "
          f"dit proces: {len(episodes)} (offset={args.offset})", flush=True)

    if args.dry_run:
        print("[dry-run] Geen wijzigingen. Episodes:")
        for i, ep in enumerate(episodes, 1):
            print(f"  {i}. {ep['title'][:70]} | {ep['duration']}s")
        return None, []

    format_id = ensure_format(
        args.rss, feed_info["title"],
        feed_info["description"], feed_info["image"],
        args.format_id
    )

    # Converteer episodes naar uniforme item-dicts
    items = []
    for ep in episodes:
        item_id, existing_audio_url = item_exists(format_id, ep["audio_url"])
        if item_id and has_fragments(item_id):
            items.append({"_skip": True, "title": ep["title"], "item_id": item_id})
            continue
        items.append({
            "_skip": False,
            "item_id": item_id or str(uuid.uuid4()),
            "title": ep["title"],
            "source_url": ep["audio_url"],
            "audio_url": existing_audio_url,  # None als nog niet op S3
            "_ep": ep,  # originele episode-dict voor save_item
        })
    return format_id, items


def run_retry_mode(args, lang: str):
    """
    Retry-pending mode: verwerk alleen items met transcript_status='pending' uit Turso.
    Geen RSS-parsing nodig — alle info staat al in de DB.
    Efficiënter voor hervatten na onderbreking.
    """
    if not args.format_id:
        print("[error] --format-id is verplicht voor --retry-pending", file=sys.stderr)
        sys.exit(1)

    pending = fetch_pending_items(args.format_id, args.offset, args.limit)
    print(f"[retry] {len(pending)} pending items voor format {args.format_id}", flush=True)

    if args.dry_run:
        print("[dry-run] Zou verwerken:")
        for i, item in enumerate(pending, 1):
            s3_status = "S3 ✅" if item.get("audio_url") else "S3 ❌"
            print(f"  {i}. {item['title'][:65]} | {s3_status}")
        return args.format_id, []

    # Converteer naar uniforme item-dicts
    items = []
    for p in pending:
        items.append({
            "_skip": False,
            "item_id": p["id"],
            "title": p["title"],
            "source_url": p["source_url"],
            "audio_url": p.get("audio_url"),  # al op S3 als pod eerder crashte na upload
            "_ep": None,  # geen episode-dict nodig — item bestaat al in Turso
        })
    return args.format_id, items


def process_items(format_id: str, items: list, args, lang: str) -> dict:
    """Verwerk een lijst van item-dicts. Gedeeld door RSS-mode en retry-mode."""
    stats = {"skipped": 0, "transcribed": 0, "failed": 0}

    for i, item in enumerate(items, 1):
        title = item["title"]
        print(f"\n[{i}/{len(items)}] {title[:65]}", flush=True)

        if item.get("_skip"):
            print(f"  ↩ Al getranscribeerd, skip", flush=True)
            stats["skipped"] += 1
            continue

        item_id = item["item_id"]
        source_url = item["source_url"]
        existing_audio_url = item.get("audio_url")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw_path = os.path.join(tmp, "raw.mp3")
                norm_path = os.path.join(tmp, "normalized.mp3")

                # Stap 1: Download (altijd origineel — voor transcriptie lokaal)
                print(f"  ↓ Download {source_url[:70]}...", flush=True)
                download(source_url, raw_path)

                # Stap 2: Normalize (ffmpeg)
                print(f"  ⚡ Normalize (192k, 1ms trim, ID3v2.3)...", flush=True)
                normalize_audio(raw_path, norm_path)

                # Stap 3: S3 upload (skip als audio_url al bestaat)
                if args.skip_s3:
                    audio_url = source_url
                    print(f"  ⏭ S3 skip, audio_url = source_url", flush=True)
                elif existing_audio_url:
                    audio_url = existing_audio_url
                    print(f"  ⏭ S3 al gedaan: {audio_url[:60]}", flush=True)
                else:
                    s3_key = f"audio/formats/{format_id}/{item_id}.mp3"
                    print(f"  ↑ Upload S3: {s3_key}", flush=True)
                    audio_url = upload_to_s3(norm_path, s3_key)
                    print(f"  ✓ S3: {audio_url}", flush=True)

                # Stap 4: Item naar Turso (INSERT OR IGNORE — bestaat al bij retry)
                if item.get("_ep") is not None:
                    save_item(item_id, format_id, item["_ep"], source_url, audio_url)
                elif not existing_audio_url:
                    # Retry-mode: update audio_url als die nog niet gezet was
                    turso_execute(
                        "UPDATE items SET audio_url = ?, updated_at = ? WHERE id = ?",
                        [audio_url, _now(), item_id]
                    )

                # Stap 5: Transcribeer (lokale genormaliseerde file)
                print(f"  🎙 Transcribeer...", flush=True)
                transcript = transcribe(norm_path, language=lang, beam_size=args.beam_size)

                # Stap 6: Fragments naar Turso
                frag_count = save_fragments(item_id, audio_url, transcript)
                print(f"  ✅ {frag_count} fragments → Turso", flush=True)
                stats["transcribed"] += 1

        except Exception as e:
            import traceback
            print(f"  ❌ Fout: {e}", flush=True)
            traceback.print_exc()
            stats["failed"] += 1
            continue

    return stats


def main():
    ap = argparse.ArgumentParser(description="Stemmy batch transcriptie pipeline")
    # RSS-mode (standaard)
    ap.add_argument("--rss", default=None, help="RSS feed URL (vereist tenzij --retry-pending)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Sla eerste N afleveringen over (voor parallelle pods)")
    ap.add_argument("--limit", type=int, default=0, help="Max afleveringen (0=alles)")
    # Retry-mode
    ap.add_argument("--retry-pending", action="store_true",
                    help="Verwerk alleen pending items uit Turso (geen RSS-parsing)")
    # Gedeeld
    ap.add_argument("--format-id", default=None,
                    help="Bestaand format UUID (verplicht bij --retry-pending)")
    ap.add_argument("--language", default=None,
                    help="Taalcode voor Whisper (nl/en/auto — default: auto-detect)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="Toon wat er zou gebeuren zonder iets te schrijven")
    ap.add_argument("--skip-s3", action="store_true",
                    help="Geen S3-upload (audio_url = source_url)")
    ap.add_argument("--auto-terminate", action="store_true",
                    help="Stop RunPod Pod automatisch na afloop (vereist RUNPOD_API_KEY + RUNPOD_POD_ID)")
    args = ap.parse_args()

    # Validatie
    if not args.retry_pending and not args.rss:
        print("[error] Geef --rss op (of gebruik --retry-pending met --format-id)", file=sys.stderr)
        sys.exit(1)

    # Controleer vereiste env vars
    required = ["TURSO_URL", "TURSO_TOKEN"]
    if not args.skip_s3:
        required += ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[error] Missende env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    lang = args.language if args.language not in (None, "auto") else None

    # Kies mode
    if args.retry_pending:
        format_id, items = run_retry_mode(args, lang)
    else:
        format_id, items = run_rss_mode(args, lang)

    if format_id is None or not items:
        return  # dry-run of niets te doen

    stats = process_items(format_id, items, args, lang)

    print(f"\n{'='*50}")
    print(f"✅ Klaar: {stats['transcribed']} getranscribeerd | "
          f"{stats['skipped']} overgeslagen | {stats['failed']} mislukt")
    print(f"Format ID: {format_id}")

    # Auto-terminate RunPod Pod (verspil geen geld na afloop)
    if args.auto_terminate or os.environ.get("AUTO_TERMINATE"):
        pod_id = os.environ.get("RUNPOD_POD_ID")
        api_key = os.environ.get("RUNPOD_API_KEY")
        if pod_id and api_key:
            print(f"\n[terminate] Stopping pod {pod_id}...", flush=True)
            try:
                r = httpx.post(
                    f"https://api.runpod.io/graphql?api_key={api_key}",
                    json={"query": f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'},
                    timeout=15.0,
                )
                print(f"[terminate] Response: {r.status_code}", flush=True)
            except Exception as e:
                print(f"[terminate] Fout: {e}", flush=True)
        else:
            print("[terminate] RUNPOD_POD_ID of RUNPOD_API_KEY niet gezet — pod draait door",
                  flush=True)


if __name__ == "__main__":
    main()
