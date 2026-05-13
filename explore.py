import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")

train = pd.read_csv(DATA_DIR / "train.csv")

print(train.head())
print(train.columns)

print("num samples:", len(train))
print("num signs:", train["sign"].nunique())

print(train["sign"].value_counts().head())

row = train.iloc[0]

path = DATA_DIR / row["path"]

print("\nLoading:", path)

df = pd.read_parquet(path)

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nHead:")
print(df.head())

print("\nTypes:")
print(df["type"].value_counts())

print("\nUnique frames:")
print(df["frame"].nunique())

print("\nLandmark counts:")
print(df.groupby("type")["landmark_index"].nunique())