"""
Text-to-Speech using Sarvam AI's TTS API (best Hindi/Indian-language voices).
Docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
Returns base64-encoded audio (WAV) which we'll resample to 8kHz for Exotel.
"""

import base64
import os

import httpx

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


async def synthesize_speech(text: str, language_code: str = "hi-IN") -> bytes:
    """Convert text to speech audio bytes (raw, base64-decoded) via Sarvam AI."""
    if not text.strip():
        return b""

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": "anushka",  # natural Hindi female voice; check docs for options
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": 8000,  # match Exotel's expected sample rate
        "enable_preprocessing": True,
        "model": "bulbul:v2",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(SARVAM_TTS_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            audio_b64 = data["audios"][0]
            return base64.b64decode(audio_b64)
    except httpx.HTTPStatusError as e:
        print(f"[TTS ERROR] {e}")
        print(f"[TTS ERROR BODY] {e.response.text}")
        return b""
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return b""
