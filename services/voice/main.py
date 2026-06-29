from fastapi import FastAPI
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

app = FastAPI(title="VoiceOS AI API")

@app.get("/")
def health_check():
    return {"status": "VoiceOS AI is running! 🚀"}

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
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        return {"error": str(e)}