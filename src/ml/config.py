import torch
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "Electronics_processed.csv"
    INPUT_PATH = BASE_DIR / "data" / "Electronics_raw.csv"
    MODEL_PATH = BASE_DIR / "models" / "model.pt"
    MAPPING_PATH = BASE_DIR / "data" / "Electronics_mapping.pkl"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
