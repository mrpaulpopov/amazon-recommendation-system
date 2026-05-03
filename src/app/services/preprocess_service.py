from src.ml.preprocess_data import preprocess
import logging


def preprocess_service(Config, samplesize):
    if not Config.INPUT_PATH.exists():
        logging.error("Dataset not found")
        raise ValueError("You should download the dataset first!")
    return preprocess(Config, samplesize)