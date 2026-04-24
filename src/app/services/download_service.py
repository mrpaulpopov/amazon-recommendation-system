import requests
from pathlib import Path
from src.ml.config import Config

def download_service(cfg: Config):
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics.csv"

    cfg.input_path.parent.mkdir(exist_ok=True)
    if not cfg.input_path.exists():
        print('downloading dataset ...')
        with requests.get(url, stream=True) as r:
            with open(cfg.input_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print('downloading complete!')