from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
from pathlib import Path
import os

# Pehle .env load karo, exotel_handler import karne se PEHLE
ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

from groq import Groq
from exotel_handler import handle_exotel_stream

app = FastAPI(title="VoiceOS AI API")

@app.get("/")
def health_check():
    return {"status": "VoiceOS AI is running! 🚀"}


@app.websocket("/ws/exotel")
async def exotel_websocket(websocket: WebSocket):
    """
    Point your Exotel Voicebot Applet at: wss://<your-domain>/ws/exotel
    (use ngrok or similar while testing locally — Exotel needs a public URL)
    """
    await handle_exotel_stream(websocket)


@app.get("/api/chat")
async def chat(message: str = "Namaste!"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY missing!"}

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Tu VoiceOS AI hai. Hindi mein baat kar."
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )
        return {
            "ai_response": response.choices[0].message.content,
            "status": "All systems ready for calls! 📞"
        }

    except Exception as e:
        return {"error": str(e)}