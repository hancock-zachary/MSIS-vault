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


# nomic-embed-text is trained with task prefixes and underperforms without them.
# Applied only on the Ollama path — OpenAI embedding models don't use prefixes.
# Prefixes exist only in the embedding request; stored chunk text stays clean.
_TASK_PREFIXES = {"document": "search_document: ", "query": "search_query: "}


def _apply_prefix(texts: list[str], kind: str) -> list[str]:
    prefix = _TASK_PREFIXES[kind]
    return [prefix + t for t in texts]


def embed_text(text: str, kind: str = "query") -> list[float]:
    if config.EMBED_PROVIDER == "ollama":
        return _ollama_embed_batch(_apply_prefix([text], kind))[0]
    return _openai_embed([text])[0]


def embed_batch(texts: list[str], kind: str = "document") -> list[list[float]]:
    if config.EMBED_PROVIDER == "ollama":
        return _ollama_embed_batch(_apply_prefix(texts, kind))
    return _openai_embed(texts)


_OLLAMA_BATCH_SIZE = 32  # Ollama rejects very large batches; send in chunks


def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    results = []
    for i in range(0, len(texts), _OLLAMA_BATCH_SIZE):
        batch = texts[i:i + _OLLAMA_BATCH_SIZE]
        resp = requests.post(config.OLLAMA_URL, json={"model": config.OLLAMA_MODEL, "input": batch})
        resp.raise_for_status()
        results.extend(resp.json()["embeddings"])
    return results


def _openai_embed(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
