import os
import subprocess
import zipfile

import duckdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

from qr_decomposition.async_typer import AsyncTyper
from qr_decomposition.device import select_device
from qr_decomposition.embedding import embed_documents
from qr_decomposition.sentiment.train import train_sentiment_model
from qr_decomposition.wikidata.train import train_wikidata_classification_model
from qr_decomposition.utils import ensure_directories, resolve_path

load_dotenv()

app = AsyncTyper(no_args_is_help=True)


def generate_labels(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df["sentiment"])  # type: ignore


@app.command("prepare-imdb")
async def prepare_dataset():
    ensure_directories(["target", "dataset"])
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

    imdb_reviews_path = resolve_path("target/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

    df = duckdb.sql(f"SELECT * FROM '{imdb_reviews_path}'").to_df()  # type: ignore
    df["labels"] = generate_labels(df)
    documents: list[str] = df["review"].tolist()
    df["embeddings"] = embed_documents(documents)

    df["split"] = np.random.choice(["train", "val"], p=[0.8, 0.2], size=len(df))

    dataset_file_path = resolve_path("dataset/imdb_reviews.parquet")
    duckdb.sql("SELECT * FROM df").write_parquet(dataset_file_path)  # type: ignore


@app.command("prepare-jigsaw-wiki")
async def prepare_jigsaw_wikidata():
    ensure_directories(["target", "dataset"])
    target_dir = resolve_path(["target", "jigsaw-wikidata"])
    # download the dataset to the target directory

    if not os.path.exists(target_dir + ".tar.xz"):
        response = requests.get(
            "https://github.com/nishtahir/such-toxic/raw/refs/heads/main/datasets/wikidata_train.csv.tar.xz"
        )
        with open(target_dir + ".tar.xz", "wb") as f:
            f.write(response.content)

    #  this was weirdly compressed, I was only able to extract
    #  it using the following command:
    #  tar -xvf wikidata_train.csv.tar.xz -C target/jigsaw-wikidata
    subprocess.run(["tar", "-xvf", target_dir + ".tar.xz", "-C", os.path.dirname(target_dir)])
    wiki_path = resolve_path(["target", "datasets", "wikidata_train.csv"])

    #  read the csv file
    df = pd.read_csv(wiki_path)  # type: ignore
    comment_text: list[str] = df["comment_text"].tolist()
    df["embeddings"] = embed_documents(comment_text)
    df["split"] = np.random.choice(["train", "val"], p=[0.8, 0.2], size=len(df))

    dataset_file_path = resolve_path("dataset/jigsaw_wikidata.parquet")
    duckdb.sql("SELECT * FROM df").write_parquet(dataset_file_path)  # type: ignore


@app.command("train-imdb-model")
async def train_imdb_model(
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


@app.command("train-wikidata-model")
async def train_wikidata_model(
    hidden_dim: int = 512,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    num_epochs: int = 10,
    dropout: float = 0.2,
    device: str = "auto",
    target_dir: str = "target/wikidata_model",
):
    ensure_directories([target_dir])

    resolved_device = select_device(device)  # type: ignore
    dataset_file_path = resolve_path("dataset/jigsaw_wikidata.parquet")
    df = duckdb.sql(f"SELECT * FROM '{dataset_file_path}'").to_df()  # type: ignore

    input_dim: int = df["embeddings"].iloc[0].shape[0]  # type: ignore

    # 6 columns in the dataframe
    # toxic, severe_toxic, obscene, threat, insult, identity_hate
    output_dim: int = 6

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")

    train_wikidata_classification_model(
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
