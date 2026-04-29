from src.ml.predict import recommend_for_user
from src.ml.config import Config

def topk_service(Config, user_str, top_k):
    if not Config.MODEL_PATH.exists():
        raise ValueError("You should train the model first!")
    return recommend_for_user(Config, user_str, top_k)