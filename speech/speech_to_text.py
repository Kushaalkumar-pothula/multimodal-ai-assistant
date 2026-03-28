"""Audio transcription utilities using Hugging Face Inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

AudioPath = Union[str, Path]


@dataclass
class AudioTranscriptionResult:
    """Structured result returned by the audio transcription module."""

    text: str
    model_id: str


class VoxtralSpeechToText:
    """Wrapper around Voxtral Mini 4B Realtime via Hugging Face Inference API."""

    DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

        if not self.api_key:
            raise ValueError(
                "Missing Hugging Face API key. Set `HUGGINGFACE_API_KEY` (or `HF_TOKEN`) in your environment."
            )

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )

    def transcribe(self, audio_path: AudioPath, prompt: Optional[str] = None) -> AudioTranscriptionResult:
        """
        Transcribe an audio file into text using Hugging Face Inference API.

        Args:
            audio_path: Path to a local audio file.
            prompt: Optional prompt for models/providers that support prompting.

        Returns:
            AudioTranscriptionResult with extracted text.
        """

        normalized_path = Path(audio_path).expanduser().resolve()

        if not normalized_path.exists():
            raise FileNotFoundError(f"Audio file not found: {normalized_path}")

        # `automatic_speech_recognition` accepts bytes/path depending on client version.
        # We pass bytes for compatibility and explicitness.
        with normalized_path.open("rb") as audio_file:
            audio_bytes = audio_file.read()

        response = self.client.automatic_speech_recognition(
            audio=audio_bytes,
            model=self.model_id,
            extra_body={"prompt": prompt} if prompt else None,
        )

        text = _extract_transcription_text(response)

        return AudioTranscriptionResult(text=text, model_id=self.model_id)


def _extract_transcription_text(response: object) -> str:
    """Normalizes Inference API output into a plain transcript string."""

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        if "text" in response and isinstance(response["text"], str):
            return response["text"].strip()
        if "generated_text" in response and isinstance(response["generated_text"], str):
            return response["generated_text"].strip()

    raise ValueError(f"Unexpected transcription response format: {response}")


def transcribe_audio_file(audio_path: AudioPath, prompt: Optional[str] = None) -> str:
    """Convenience helper used by the assistant pipeline."""

    return VoxtralSpeechToText().transcribe(audio_path=audio_path, prompt=prompt).text
