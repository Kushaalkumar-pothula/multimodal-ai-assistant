# multimodal-ai-assistant

Multimodal AI assistant with vision, speech, memory and agent capabilities.

## Audio integration (Voxtral Mini 4B Realtime)

This repository includes audio transcription via the Hugging Face Inference API:

- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- Runtime: hosted inference (no local model weights required)
- Auth: `HUGGINGFACE_API_KEY` (or `HF_TOKEN`)

### How audio is wired into the assistant pipeline

1. `process_user_query(...)` accepts:
   - `audio_data` (already-transcribed text), or
   - `audio_file_path` (local audio file path).
2. If `audio_file_path` is provided without `audio_data`, the assistant calls Hugging Face Inference API to transcribe with Voxtral.
3. The transcript is injected into context as `Detected Audio Transcript`.
4. The final LLM answer is generated from vision + audio + memory + conversation context.

### Environment variables

Create a `.env` file with both keys used by this project:

```bash
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key
```

### Example usage

```python
from brain.assistant import process_user_query

result = process_user_query(
    user_input="Summarize what was said in the audio.",
    audio_file_path="/path/to/meeting_clip.wav",
    vision_data="A slide that says Q2 Planning",
    instruction="Give a brief summary"
)

print(result["audio_transcript"])
print(result["response"])
```

## Docker

A `Dockerfile` is included for a simple runnable environment. Since transcription is API-based,
no local GPU model serving is required.

Build and run:

```bash
docker build -t multimodal-ai-assistant .
docker run --rm -it --env-file .env multimodal-ai-assistant
```
