"""
groq_client.py — Groq API integration with LLaMA 3.3 70B
"""
import os
from groq import Groq
from typing import List, Dict

client = None


def get_client() -> Groq:
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")
        client = Groq(api_key=api_key)
    return client


def build_system_prompt(context: str) -> str:
    if context:
        return f"""You are an intelligent RAG assistant. Answer the user's question strictly using the context below.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."
Always mention which source(s) you used in your answer.

RETRIEVED CONTEXT:
{context}
"""
    return (
        "You are a helpful assistant. No documents have been uploaded yet. "
        "Answer from general knowledge and remind the user to upload documents for RAG."
    )


def chat(
    messages: List[Dict[str, str]],
    context: str = "",
    stream: bool = True
):
    """
    Send messages to Groq LLaMA and return response.
    If stream=True, yields text chunks.
    """
    model = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
    system_prompt = build_system_prompt(context)

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    groq = get_client()

    if stream:
        response = groq.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1024,
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    else:
        response = groq.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1024,
            temperature=0.2,
        )
        yield response.choices[0].message.content
