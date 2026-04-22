from src.ml.config import Config
from src.ml.train import train_pipeline

def train_model(cfg: Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS):
    if not cfg.data_path.exists() or not cfg.mapping_path.exists():
        raise ValueError("You should preprocess your data first!")
    return train_pipeline(cfg, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS)