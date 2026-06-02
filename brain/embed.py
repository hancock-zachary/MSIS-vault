import os
import requests
from brain.config import EMBED_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, OPENAI_EMBED_MODEL


def embed_text(text: str) -> list[float]:
    if EMBED_PROVIDER == "ollama":
        return _ollama_embed(text)
    return _openai_embed([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    if EMBED_PROVIDER == "ollama":
        return [_ollama_embed(t) for t in texts]
    return _openai_embed(texts)


def _ollama_embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
