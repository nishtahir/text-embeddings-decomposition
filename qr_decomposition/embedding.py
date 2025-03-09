import os

import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: list[str], model: str = "text-embedding-3-small") -> npt.NDArray[np.float32]:
    res = oai.embeddings.create(input=text, model=model).data
    embs = [item.embedding for item in res]
    return np.array(embs)
