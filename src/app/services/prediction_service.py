from src.ml.predict import predict_user_items
from src.ml.config import Config

def predict_service(cfg: Config, user_str, item_strs):
    if not cfg.model_path.exists():
        raise ValueError("You should train the model first!")
    return predict_user_items(cfg, user_str, item_strs)