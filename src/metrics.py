"""
Dos familias de métricas:

1. Distancias vectoriales "manuales" (independientes de Chroma) — sirven
   para mostrar en el dashboard, de forma didáctica, la fórmula exacta que
   hay detrás de cada uno de los 3 métodos, y para verificar que el score
   que reporta Chroma es consistente con el cálculo directo.

2. Métricas de calidad de recuperación (precision@k, recall@k, MRR) contra
   un dataset de referencia (eval/eval_dataset.json), calculadas para cada
   combinación (métrica de distancia × estrategia de chunking).
"""
import json

import numpy as np


# ---------- 1. Distancias vectoriales ----------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return 1.0 - sim


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def dot_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Chroma reporta el producto punto negado para que "menor = más cercano",
    # igual que las otras dos métricas.
    return -float(np.dot(a.astype(np.float64), b.astype(np.float64)))


DISTANCE_FUNCS = {
    "cosine": cosine_distance,
    "l2": euclidean_distance,
    "ip": dot_product_distance,
}


# ---------- 2. Precision / Recall / MRR de recuperación ----------

def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(query_all_distances_fn, k: int = 5, chunk_strategy: str | None = None,
                        eval_path: str = "eval/eval_dataset.json") -> "pandas.DataFrame":
    """Corre cada pregunta del eval set contra las 3 métricas de distancia y
    devuelve un DataFrame con precision@k, recall@k y MRR por métrica.

    El dataset de eval usa `source` + `heading`/fragmento de texto como
    criterio de relevancia (ver eval/eval_dataset.json) en vez de IDs de
    chunk fijos, porque los IDs se regeneran en cada reindexado.
    """
    import pandas as pd

    dataset = load_eval_dataset(eval_path)
    rows = []

    for item in dataset:
        query = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = [kw.lower() for kw in item.get("expected_keywords", [])]

        results = query_all_distances_fn(query, k=k, chunk_strategy=chunk_strategy)

        for distance, res in results.items():
            retrieved_sources = [m["source"] for m in res["metadatas"]]
            retrieved_texts = [d.lower() for d in res["documents"]]

            # Un chunk se considera "relevante" si viene del documento
            # esperado Y contiene al menos una de las palabras clave
            # esperadas — aproximación simple pero verificable a mano.
            relevant_flags = [
                (src == expected_source) and any(kw in txt for kw in expected_keywords)
                for src, txt in zip(retrieved_sources, retrieved_texts)
            ]
            retrieved_ids = [f"pos_{i}" if flag else f"neg_{i}" for i, flag in enumerate(relevant_flags)]
            relevant_ids = {f"pos_{i}" for i, flag in enumerate(relevant_flags) if flag}

            rows.append(
                {
                    "question": query,
                    "distance": distance,
                    "chunk_strategy": chunk_strategy or "mixto",
                    f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
                    f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
                    "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                }
            )

    return pd.DataFrame(rows)
