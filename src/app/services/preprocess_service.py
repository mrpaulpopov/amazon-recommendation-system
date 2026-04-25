from src.ml.config import Config
from src.ml.preprocess_data import preprocess

def preprocess_service(Config, samplesize):
    if not Config.INPUT_PATH.exists():
        raise ValueError("You should download the dataset first!")
    return preprocess(Config, samplesize)