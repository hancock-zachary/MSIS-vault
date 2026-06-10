from unittest.mock import patch, MagicMock
import src.config as brain_config
from src.embed import embed_text, embed_batch


def test_embed_text_returns_list_of_floats():
    fake_vector = [0.1] * 768
    with patch("src.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [fake_vector]},
            raise_for_status=lambda: None,
        )
        result = embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_list_of_vectors():
    fake_vector = [0.1] * 768
    with patch("src.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [fake_vector, fake_vector]},
            raise_for_status=lambda: None,
        )
        results = embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 768


def test_embed_batch_prepends_search_document_prefix_for_ollama():
    fake_vector = [0.1] * 768
    with patch("src.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [fake_vector, fake_vector]},
            raise_for_status=lambda: None,
        )
        embed_batch(["hello", "world"])
    sent = mock_post.call_args.kwargs["json"]["input"]
    assert sent == ["search_document: hello", "search_document: world"]


def test_embed_text_prepends_search_query_prefix_for_ollama():
    fake_vector = [0.1] * 768
    with patch("src.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [fake_vector]},
            raise_for_status=lambda: None,
        )
        embed_text("what is an ERD?")
    sent = mock_post.call_args.kwargs["json"]["input"]
    assert sent == ["search_query: what is an ERD?"]


def test_embed_batch_kind_query_uses_query_prefix():
    fake_vector = [0.1] * 768
    with patch("src.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [fake_vector]},
            raise_for_status=lambda: None,
        )
        embed_batch(["what is an ERD?"], kind="query")
    sent = mock_post.call_args.kwargs["json"]["input"]
    assert sent == ["search_query: what is an ERD?"]


def test_openai_path_sends_text_without_prefix():
    fake_vector = [0.2] * 1536
    mock_embedding = MagicMock()
    mock_embedding.embedding = fake_vector
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    with patch.object(brain_config, "EMBED_PROVIDER", "openai"):
        with patch("src.embed._get_openai_client") as mock_client_getter:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_client_getter.return_value = mock_client
            embed_text("hello world")

    sent = mock_client.embeddings.create.call_args.kwargs["input"]
    assert sent == ["hello world"]


def test_embed_text_openai_path():
    fake_vector = [0.2] * 1536  # OpenAI embedding size
    mock_embedding = MagicMock()
    mock_embedding.embedding = fake_vector
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]

    with patch.object(brain_config, "EMBED_PROVIDER", "openai"):
        with patch("src.embed._get_openai_client") as mock_client_getter:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_client_getter.return_value = mock_client
            result = embed_text("hello world")

    assert isinstance(result, list)
    assert len(result) == 1536
