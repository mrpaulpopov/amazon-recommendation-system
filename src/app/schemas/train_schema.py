from pydantic import BaseModel

class TrainModel(BaseModel):
    n_epoch: int = 70
    learning_rate: float = 0.001
    batch_size: int = 128
    embedding_dim: int = 64