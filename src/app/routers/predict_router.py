from fastapi import APIRouter, Depends, HTTPException
from src.app.schemas.predict_schema import PredictModel
from src.app.dependencies.security import verify_api_key
from src.app.services.prediction_service import predict_service
from src.ml.config import Config
import logging
import time

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/")
def predict_endpoint(data: PredictModel, api_key: str = Depends(verify_api_key)):
    logging.info("Prediction request received")
    start = time.time()
    try:
        result = predict_service(Config, user_str=data.user_str, item_strs=data.item_strs)
        logging.info(f"Prediction completed in {time.time() - start:.4f}s")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction" : result}
