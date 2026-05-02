from src.ml.predict import predict_user_items
from fastapi import HTTPException
from src.ml.config import Config
import logging

def predict_service(Config, user_str, item_strs):
    if not Config.MODEL_PATH.exists():
        logging.error("Model not found")
        raise HTTPException(status_code=503, detail="Model not loaded, you should train the model first!")
    return predict_user_items(Config, user_str, item_strs)