import requests
from pathlib import Path
from src.ml.config import Config

def download_service(Config):
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics.csv"

    Config.INPUT_PATH.parent.mkdir(exist_ok=True)
    if not Config.INPUT_PATH.exists():
        print('downloading dataset ...')
        with requests.get(url, stream=True) as r:
            with open(Config.INPUT_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print('downloading complete!')