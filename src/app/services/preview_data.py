import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PATH = BASE_DIR / "data" / "Electronics_processed.csv"

df = pd.read_csv(PATH)

print(df.head())
print('----------------')
print(df.describe()) # быстрый показ статистических данных
print('----------------')
print(df.info()) # понять структуру таблицы