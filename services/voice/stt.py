"""
Speech-to-Text using Groq's Whisper endpoint.
Exotel sends raw audio chunks (base64, 8kHz mono PCM/slin or wav depending on config).
We buffer chunks per call and transcribe on silence/turn-end.
"""

import io
import os
import wave

from groq import AsyncGroq

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw PCM bytes in a WAV header so Whisper can read it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def transcribe_audio(pcm_data: bytes, sample_rate: int = 8000) -> str:
    """Send buffered audio to Groq Whisper and return transcript text."""
    if not pcm_data:
        return ""

    wav_bytes = pcm_to_wav_bytes(pcm_data, sample_rate)

    try:
        result = await groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model="whisper-large-v3-turbo",
            language="hi",  # Hindi; change/remove for auto-detect
            response_format="text",
        )
        return str(result).strip()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return ""
