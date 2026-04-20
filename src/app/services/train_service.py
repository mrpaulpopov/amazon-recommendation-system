from src.ml.train import train_pipeline

def train_model(Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS):
    return train_pipeline(Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS)