from fastapi import FastAPI

from src.app.routers import predict_router, train_router

app = FastAPI()

app.include_router(predict_router.router)

app.include_router(train_router.router)
# uvicorn src.app.main:app --reload
