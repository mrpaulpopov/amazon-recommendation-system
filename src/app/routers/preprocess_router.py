from fastapi import APIRouter, Depends, HTTPException
from src.app.schemas.preprocess_schema import PreprocessData
from src.app.dependencies.security import verify_api_key
from src.app.services.preprocess_service import preprocess_service
from src.ml.config import Config

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.post("/")
def preprocess_endpoint(data: PreprocessData, api_key: str = Depends(verify_api_key)):
    try:
        preprocess_service(Config, samplesize=data.samplesize)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"Preprocessing finished."}
