"""
Cliente para Groq: LLM vía SDK OpenAI-compatible + embeddings
vía sentence-transformers local (Groq no tiene endpoint de embeddings,
se usan embeddings locales gratuitos que corren en el servidor de Streamlit).
"""
import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embedding_model():
  from sentence_transformers import SentenceTransformer
  return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
  if not GROQ_API_KEY:
    raise RuntimeError(
            "Falta GROQ_API_KEY. Agrégala en los Secrets de Streamlit Cloud."
      )
  return OpenAI(base_url=GROQ_ENDPOINT, api_key=GROQ_API_KEY)


def embed_texts(texts: list[str]) -> np.ndarray:
  """Embeddings locales con sentence-transformers (gratuito, sin API externa)."""
  model = get_embedding_model()
  vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
  return np.array(vectors, dtype=np.float32)



def embed_query(text: str) -> np.ndarray:
  return embed_texts([text])[0]


def chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
  client = get_chat_client()
  resp = client.chat.completions.create(
  model=CHAT_MODEL,
  temperature=temperature,
  messages=[
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": user_prompt},
  ],
  )
  return resp.choices[0].message.content

