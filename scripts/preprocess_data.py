import argparse
from src.data.preprocess import preprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int)
arg = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "Electronics_raw.csv"
OUTPUT_PATH = BASE_DIR / "data" / "Electronics_processed.csv"
MAPPING_PATH = BASE_DIR / "data" / "Electronics_mapping.pkl"

if arg.sample is not None:
    preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH, mapping_path=MAPPING_PATH, samplesize=arg.sample)
else:
    preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH, mapping_path=MAPPING_PATH)

# PYTHONPATH=. python scripts/preprocess_data.py --sample 100000