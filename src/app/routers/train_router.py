from fastapi import APIRouter, Depends, HTTPException
from src.app.schemas.train_schema import TrainModel
from src.app.dependencies.security import verify_api_key
from src.app.services.train_service import train_model
from src.ml.config import Config
import logging
import time

router = APIRouter(prefix="/train", tags=["train"])


@router.post("/")
def train_endpoint(data: TrainModel, api_key: str = Depends(verify_api_key)):
    logging.info("Train request received")
    start = time.time()
    try:
        result = train_model(Config, EMBEDDING_DIM=data.embedding_dim,
                             LEARNING_RATE=data.learning_rate, BATCH_SIZE=data.batch_size, N_EPOCHS=data.n_epoch)
        logging.info(f"Training completed in {time.time() - start:.4f}s")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction:" : result}
