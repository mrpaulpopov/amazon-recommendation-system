from fastapi import HTTPException
from src.ml.predict import recommend_for_user
import logging

def topk_service(Config, user_str, top_k):
    if not Config.MODEL_PATH.exists():
        logging.error("Model not found")
        raise HTTPException(status_code=503, detail="Model not loaded, you should train the model first!")
    return recommend_for_user(Config, user_str, top_k)