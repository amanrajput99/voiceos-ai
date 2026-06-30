"""
Conversation brain — takes transcript + history, returns the agent's reply text.
Later: load Agent.prompt from DB (per-company customization) instead of static SYSTEM_PROMPT.
"""

import os

from groq import AsyncGroq

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """Tum VoiceOS AI ho — ek company ke liye phone par customers se baat karne wala
AI voice agent. Hindi mein, dosti aur professional tone mein, chhote aur clear jawab do
(kyunki ye बोला jaayega, lamba paragraph mat likho). Agar customer ka sawal samajh na aaye,
politely dobara puchho."""


async def get_agent_reply(
    user_text: str,
    conversation_history: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Get a short, voice-friendly reply from the LLM."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=150,  # keep replies short — this is a phone call, not an essay
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "Maaf kijiye, mujhe samajhne mein dikkat ho rahi hai. Kya aap dobara bol sakte hain?"
