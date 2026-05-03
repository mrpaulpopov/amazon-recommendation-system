from fastapi import APIRouter, HTTPException
from src.app.services.download_service import download_service
from src.ml.config import Config

router = APIRouter(prefix="/download", tags=["download"])


@router.get("/")
def download_endpoint():
    try:
        download_service(Config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"Preprocessing finished."}
