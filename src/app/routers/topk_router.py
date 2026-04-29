from fastapi import APIRouter, Depends, HTTPException
from src.app.schemas.topk_schema import TopkModel
from src.app.dependencies.security import verify_api_key
from src.app.services.topk_service import topk_service
from src.ml.config import Config

router = APIRouter(prefix="/topk", tags=["topk"])


@router.post("/")
def predict_endpoint(data: TopkModel, api_key: str = Depends(verify_api_key)):
    try:
        result = topk_service(Config, user_str=data.user_str, top_k=data.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction:" : result}
