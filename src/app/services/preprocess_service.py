from src.ml.config import Config
from src.ml.preprocess_data import preprocess

def preprocess_service(cfg: Config, samplesize):
    if not cfg.input_path.exists():
        raise ValueError("You should download the dataset first!")
    return preprocess(cfg, samplesize)