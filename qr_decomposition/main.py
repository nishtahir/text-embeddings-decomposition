import os
import zipfile

import duckdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from qr_decomposition.async_typer import AsyncTyper
from qr_decomposition.device import select_device
from qr_decomposition.embedding import embed_text
from qr_decomposition.sentiment.train import train_sentiment_model
from qr_decomposition.utils import ensure_directories, resolve_path

load_dotenv()

app = AsyncTyper(no_args_is_help=True)


def download_dataset():
    target_dir = resolve_path("target/imdb-dataset-of-50k-movie-reviews")
    if not os.path.exists(target_dir):
        # download the dataset to the target directory
        response = requests.get(
            "https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        )
        with open(target_dir + ".zip", "wb") as f:
            f.write(response.content)

    with zipfile.ZipFile(target_dir + ".zip", "r") as zip_ref:
        zip_ref.extractall(target_dir)


def generate_labels(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df["sentiment"])  # type: ignore


def embed_reviews(all_reviews: list[str], batch_size: int = 1000) -> list[float]:
    embedded_reviews: list[float] = []
    for i in tqdm(range(0, len(all_reviews), batch_size)):
        reviews = all_reviews[i : i + batch_size]
        text_embeddings = embed_text(reviews)
        embedded_reviews.extend(text_embeddings)

    return embedded_reviews


@app.command("prepare-dataset")
async def prepare_dataset():
    ensure_directories(["target", "dataset"])
    download_dataset()

    imdb_reviews_path = resolve_path("target/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

    df = duckdb.sql(f"SELECT * FROM '{imdb_reviews_path}'").to_df()  # type: ignore
    df["labels"] = generate_labels(df)
    reviews: list[str] = df["review"].tolist()
    df["embeddings"] = embed_reviews(reviews)

    df["split"] = np.random.choice(["train", "val"], p=[0.8, 0.2], size=len(df))

    dataset_file_path = resolve_path("dataset/imdb_reviews.parquet")
    duckdb.sql("SELECT * FROM df").write_parquet(dataset_file_path)  # type: ignore


@app.command("train-model")
async def train_model(
    hidden_dim: int = 256,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 10,
    dropout: float = 0.3,
    device: str = "auto",
    target_dir: str = "target/sentiment_model",
):
    ensure_directories([target_dir])

    resolved_device = select_device(device)  # type: ignore
    dataset_file_path = resolve_path("dataset/imdb_reviews.parquet")
    df = duckdb.sql(f"SELECT * FROM '{dataset_file_path}'").to_df()  # type: ignore

    input_dim: int = df["embeddings"].iloc[0].shape[0]  # type: ignore
    output_dim: int = len(df["labels"].unique())  # type: ignore

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")

    train_sentiment_model(
        input_dim=input_dim,  # type: ignore
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=resolved_device,
        train_df=train_df,
        val_df=val_df,
        target_dir=target_dir,
    )


if __name__ == "__main__":
    app()
