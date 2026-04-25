from src.ml.predict import predict_user_items
from src.ml.config import Config

def predict_service(Config, user_str, item_strs):
    if not Config.MODEL_PATH.exists():
        raise ValueError("You should train the model first!")
    return predict_user_items(Config, user_str, item_strs)