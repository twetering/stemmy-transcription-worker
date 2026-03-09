# Stemmy Transcription Worker (RunPod Serverless)

WhisperX transcription worker voor RunPod Serverless. Converteert audio naar AssemblyAI-compatibel JSON.

## Deploy op RunPod

1. **GitHub**: Push deze repo naar je GitHub (bijv. `twetering/stemmy-transcription-worker`)
2. **RunPod**: Serverless → Deploy Endpoint → Import GitHub Repository
3. **Selecteer** deze repo, branch `main`
4. **GPU**: Kies RTX 4090 of L4
5. **Deploy** – RunPod bouwt de image automatisch

## Input / Output

**Input** (per request):
```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "item_id": "uuid-optional",
    "batch_size": 16
  }
}
```

**Output** (AssemblyAI-compatibel):
```json
{
  "text": "...",
  "words": [{"text": "...", "start": 0, "end": 100, "speaker": "A", "confidence": 0.9}],
  "utterances": [...],
  "language_code": "nl",
  "item_id": "uuid"
}
```

## Lokaal testen (met GPU)

```bash
pip install -r requirements.txt runpod
python rp_handler.py --test_input '{"input": {"audio_url": "https://..."}}'
```

## Gebruik met orchestrate_batch

Voeg `RUNPOD_ENDPOINT_ID` toe aan `.env` in surrounded, dan:

```bash
cd surrounded
python scripts/orchestrate_batch.py --limit 10
```
