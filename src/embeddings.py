"""
Cliente para GitHub Models: mismo catálogo de modelos que usa GitHub Copilot,
expuesto vía un endpoint compatible con el SDK de OpenAI. Se autentica con
un GITHUB_TOKEN (gratis, con cuota diaria) en vez de una API key de OpenAI.

Doc del catálogo: https://github.com/marketplace/models
"""
import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ENDPOINT = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Falta GITHUB_TOKEN. Crea un Personal Access Token en GitHub "
            "(Settings > Developer settings > Personal access tokens) y "
            "colócalo en tu .env o en los Secrets de Streamlit Cloud."
        )
    return OpenAI(base_url=ENDPOINT, api_key=GITHUB_TOKEN)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Genera embeddings para una lista de textos. Devuelve un array (n, d)."""
    client = get_client()
    # El endpoint acepta batches; se trocea por seguridad ante límites de payload.
    vectors = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
