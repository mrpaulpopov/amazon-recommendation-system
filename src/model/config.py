# import torch
# from pathlib import Path
#
# class Config:
#     EMBEDDING_DIM = 64
#     N_EPOCHS = 10
#     LEARNING_RATE = 0.01
#     BATCH_SIZE = 128
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     data_path = BASE_DIR / "data" / "Electronics_processed.csv"
#     model_path = BASE_DIR / "models" / "model.pt"
#     mapping_path = BASE_DIR / "data" / "Electronics_mapping.pkl"