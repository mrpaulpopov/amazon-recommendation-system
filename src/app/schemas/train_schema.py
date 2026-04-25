from pydantic import BaseModel

class TrainModel(BaseModel):
    n_epoch: int = 17
    learning_rate: float = 0.0005
    batch_size: int = 128
    embedding_dim: int = 128