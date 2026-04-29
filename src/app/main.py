from fastapi import FastAPI
from src.app.routers import predict_router, train_router, preprocess_router, download_router, topk_router

app = FastAPI()

app.include_router(predict_router.router)

app.include_router(train_router.router)

app.include_router(preprocess_router.router)

app.include_router(download_router.router)

app.include_router(topk_router.router)
# uvicorn src.app.main:app --reload
