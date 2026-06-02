from unittest.mock import patch, MagicMock
from brain.embed import embed_text, embed_batch


def test_embed_text_returns_list_of_floats():
    fake_vector = [0.1] * 768
    with patch("brain.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embedding": fake_vector},
            raise_for_status=lambda: None,
        )
        result = embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_list_of_vectors():
    fake_vector = [0.1] * 768
    with patch("brain.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embedding": fake_vector},
            raise_for_status=lambda: None,
        )
        results = embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 768
