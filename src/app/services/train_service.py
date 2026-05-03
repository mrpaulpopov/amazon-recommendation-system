from src.ml.train import train_pipeline
import logging

def train_model(Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS):
    if not Config.DATA_PATH.exists() or not Config.MAPPING_PATH.exists():
        logging.error("Preprocessed file or mapping file not found")
        raise ValueError("You should preprocess your data first!")
    return train_pipeline(Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS)