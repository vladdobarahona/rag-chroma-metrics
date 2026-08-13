"""
Cliente para GitHub Models: mismo catálogo de modelos que usa GitHub Copilot,
expuesto vía un endpoint compatible con el SDK de OpenAI. Se autentica con
un GITHUB_TOKEN (gratis, con cuota diaria) en vez de una API key de OpenAI.

Doc del catálogo: https://github.com/marketplace/models
"""
import os
from functools import lru_cache

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ENDPOINT = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Falta GITHUB_TOKEN. Agrégalo en los Secrets de Streamlit Cloud."
        )
    return OpenAI(base_url=ENDPOINT, api_key=GITHUB_TOKEN)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embeddings via requests directo al endpoint REST de GitHub Models."""
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Falta GITHUB_TOKEN"
        )
    headers ={"Authorization": f"Bearer {GITHUB_TOKEN}",
              "Content-Type": "application/json",
              "Accept":"application/vnd.github+json",
              "X-GitHub-Api-Version":"2026-03-10",}    
    
    vectors = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(f"{ENDPOINT}/embeddings",
                            headers=headers,
                            json={"model":EMBEDDING_MODEL,"input":batch},
                            timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Error {resp.status_code} embeddings: {resp.text[:400]}"
                              )
        items =sorted(resp.json()["data"], key = lambda d: d["index"])
        vectors.extend([d.embedding for d in items])
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
