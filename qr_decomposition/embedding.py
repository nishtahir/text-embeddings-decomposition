import os

import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: list[str], model: str = "text-embedding-3-small") -> npt.NDArray[np.float32]:
    res = oai.embeddings.create(input=text, model=model).data
    embs = [item.embedding for item in res]
    return np.array(embs)


def embed_documents(documents: list[str], batch_size: int = 1000) -> list[float]:
    embedded_documents: list[float] = []
    for i in tqdm(range(0, len(documents), batch_size)):
        documents = documents[i : i + batch_size]
        text_embeddings = embed_text(documents)
        embedded_documents.extend(text_embeddings)

    return embedded_documents
