from dataclasses import dataclass
import torch
from pathlib import Path


@dataclass
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    data_path = BASE_DIR / "data" / "Electronics_processed.csv"
    model_path = BASE_DIR / "models" / "model.pt"
    mapping_path = BASE_DIR / "data" / "Electronics_mapping.pkl"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
