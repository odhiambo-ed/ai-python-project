import pytest
import numpy as np
from src.data.processor import read_jsonl, filter_high_quality, count_records
from src.models.schemas import ChatMessage, ChatRequest
from src.utils.data_science import cosine_similarity, softmax


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 1.0

    c = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, c) == 0.0


def test_softmax():
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    assert len(probs) == 3
    assert abs(probs.sum() - 1.0) < 1e-6


def test_chat_message_validation():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"

    with pytest.raises(Exception):
        ChatMessage(role="invalid", content="Hello")


def test_data_processing(tmp_path):
    # Create test JSONL file
    test_data = [
        {"id": "1", "text": "Good", "quality_score": 0.9},
        {"id": "2", "text": "Bad", "quality_score": 0.3},
    ]

    test_file = tmp_path / "test.jsonl"
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(f"{item}\n")

    # Test generator
    records = list(read_jsonl(str(test_file)))
    assert len(records) == 2

    # Test filtering
    high_quality = list(filter_high_quality(read_jsonl(str(test_file))))
    assert len(high_quality) == 1
    assert high_quality[0]["quality_score"] == 0.9
