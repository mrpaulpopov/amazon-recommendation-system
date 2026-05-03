import pytest
import torch
from fastapi.testclient import TestClient

from src.app.main import app
from src.ml.config import Config
from src.ml.model import MFModel
from src.ml.predict import predict_user_items, recommend_for_user


def test_predict_unknown_user_returns_zero():
    result = predict_user_items(Config, "unknown_user", ["item_1"])

    assert result == [0.0]


def test_recommend_returns_top_k():
    result = recommend_for_user(Config, "A3PHJ4NMHMBBUB", top_k=5)

    assert isinstance(result, list)
    assert len(result) == 5


def test_recommend_unknown_user():
    with pytest.raises(ValueError):
        recommend_for_user(Config, "unknown_user", top_k=5)


def test_predict_endpoint():
    client = TestClient(app)

    response = client.post(
        "/predict",
        headers={"x-api-key": "secret123"},
        json={"user_str": "user_1", "item_strs": ["item_1"]}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_model_forward_shape():
    model = MFModel(num_users=10, num_items=10, embedding_dim=4)

    users = torch.tensor([1, 2])
    items = torch.tensor([3, 4])

    out = model(users, items)

    assert out.shape[0] == 2
