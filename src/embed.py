import os
import requests
from src import config


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client


def embed_text(text: str) -> list[float]:
    if config.EMBED_PROVIDER == "ollama":
        return _ollama_embed(text)
    return _openai_embed([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    if config.EMBED_PROVIDER == "ollama":
        return _ollama_embed_batch(texts)
    return _openai_embed(texts)


def _ollama_embed(text: str) -> list[float]:
    return _ollama_embed_batch([text])[0]


def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(config.OLLAMA_URL, json={"model": config.OLLAMA_MODEL, "input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
