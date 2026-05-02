from fastapi import APIRouter, Depends, HTTPException
from src.app.schemas.topk_schema import TopkModel
from src.app.dependencies.security import verify_api_key
from src.app.services.topk_service import topk_service
from src.ml.config import Config
import logging
import time

router = APIRouter(prefix="/topk", tags=["topk"])


@router.post("/")
def predict_endpoint(data: TopkModel, api_key: str = Depends(verify_api_key)):
    logging.info("Prediction request received")
    start = time.time()
    try:
        result = topk_service(Config, user_str=data.user_str, top_k=data.top_k)
        logging.info(f"Prediction completed in {time.time() - start:.4f}s")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction" : result}
