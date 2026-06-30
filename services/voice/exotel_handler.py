"""
Handles the WebSocket stream Exotel's Voicebot Applet opens for each call.

Exotel sends JSON events over the socket:
  - "connected" : handshake
  - "start"     : call started, includes streamSid / callSid + audio format info
  - "media"     : { "payload": "<base64 audio chunk>" } — repeated, ~20ms chunks
  - "stop"      : call ended

We send back:
  - { "event": "media", "stream_sid": ..., "media": { "payload": "<base64>" } }
  - { "event": "mark", "stream_sid": ..., "mark": { "name": "..." } }

NOTE: verify exact field names against Exotel's current Voicebot Applet docs
(https://developer.exotel.com/) once the account is live — protocol details
can shift slightly between providers/versions. This handler is structured so
swapping field names later is a small, localized change.
"""

import asyncio
import base64
import json

from fastapi import WebSocket, WebSocketDisconnect

from agent import get_agent_reply
from stt import transcribe_audio
from tts import synthesize_speech

# How many silent/empty media frames before we treat it as "caller stopped talking"
# and trigger a transcription + reply. Tune this once testing on real calls.
SILENCE_FRAMES_THRESHOLD = 40  # roughly ~0.8s of continuous silence
MIN_AUDIO_BYTES = 9600  # ~0.6s minimum audio before transcribing

class CallSession:
    def __init__(self, stream_sid: str):
        self.stream_sid = stream_sid
        self.audio_buffer = bytearray()
        self.silence_count = 0
        self.history: list[dict] = []


async def handle_exotel_stream(websocket: WebSocket):
    await websocket.accept()
    session: CallSession | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                print("[Exotel] connected")

            elif event == "start":
                stream_sid = msg.get("stream_sid") or msg.get("streamSid")
                session = CallSession(stream_sid)
                print(f"[Exotel] call started: {stream_sid}")
                print(f"[Exotel] FULL START PAYLOAD: {json.dumps(msg)}")
                await send_greeting(websocket, session)

            elif event == "media" and session:
                payload = msg["media"]["payload"]
                chunk = base64.b64decode(payload)
                session.audio_buffer.extend(chunk)

                # Naive turn-taking: in production replace with proper VAD
                # (e.g. webrtcvad) instead of a frame counter.
                if is_silence(chunk):
                    session.silence_count += 1
                else:
                    session.silence_count = 0

                if (
                    session.silence_count >= SILENCE_FRAMES_THRESHOLD
                    and len(session.audio_buffer) >= MIN_AUDIO_BYTES
                ):
                    await process_turn(websocket, session)

            elif event == "stop":
                print("[Exotel] call ended")
                break

    except WebSocketDisconnect:
        print("[Exotel] websocket disconnected")
    except Exception as e:
        print(f"[Exotel handler ERROR] {e}")


def is_silence(chunk: bytes, threshold: int = 500) -> bool:
    """Extremely rough silence check on 16-bit PCM. Swap for webrtcvad for real use."""
    if len(chunk) < 2:
        return True
    samples = [
        int.from_bytes(chunk[i : i + 2], "little", signed=True)
        for i in range(0, len(chunk) - 1, 2)
    ]
    avg_amplitude = sum(abs(s) for s in samples) / len(samples)
    return avg_amplitude < threshold


async def process_turn(websocket: WebSocket, session: CallSession):
    pcm_data = bytes(session.audio_buffer)
    session.audio_buffer = bytearray()
    session.silence_count = 0

    user_text = await transcribe_audio(pcm_data)
    if not user_text:
        return

    print(f"[Caller said]: {user_text}")
    session.history.append({"role": "user", "content": user_text})

    reply_text = await get_agent_reply(user_text, session.history[:-1])
    session.history.append({"role": "assistant", "content": reply_text})
    print(f"[Agent reply]: {reply_text}")

    audio_bytes = await synthesize_speech(reply_text)
    await stream_audio_back(websocket, session, audio_bytes)


async def send_greeting(websocket: WebSocket, session: CallSession):
    greeting = "Namaste! Main aapka AI assistant hoon. Aap kaise madad chahte hain?"
    session.history.append({"role": "assistant", "content": greeting})
    audio_bytes = await synthesize_speech(greeting)
    await stream_audio_back(websocket, session, audio_bytes)


async def stream_audio_back(websocket: WebSocket, session: CallSession, audio_bytes: bytes):
    if not audio_bytes:
        return

    chunk_size = 320  # ~20ms of 8kHz 16-bit mono audio
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "stream_sid": session.stream_sid,
                    "media": {"payload": payload},
                }
            )
        )
        await asyncio.sleep(0.02)  # pace chunks roughly real-time

    await websocket.send_text(
        json.dumps(
            {
                "event": "mark",
                "stream_sid": session.stream_sid,
                "mark": {"name": "reply_complete"},
            }
        )
    )
