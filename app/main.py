import torch
from src.model.predict import predict_user_items
from src.model.train import train_pipeline
from pathlib import Path


EMBEDDING_DIM = 64
N_EPOCHS = 70
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "Electronics_processed.csv"
model_path = BASE_DIR / "models" / "model.pt"
mapping_path = BASE_DIR / "data" / "Electronics_mapping.pkl"

# train_pipeline(data_path, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, DEVICE, N_EPOCHS, model_path, mapping_path)

preds = predict_user_items(model_path, mapping_path, EMBEDDING_DIM, user_str="A1GYLH2VMW2MQI", item_strs=["B00NIYJL64", "B00BKVQETY", "B016I3T3CI"])
print(preds)