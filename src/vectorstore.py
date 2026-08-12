"""
Crea y consulta TRES colecciones Chroma sobre el mismo set de chunks, cada
una configurada con una métrica de distancia distinta vía el parámetro
`hnsw:space` de Chroma:

    "cosine" -> similitud coseno   (1 - similitud coseno = distancia)
    "l2"     -> distancia euclídea al cuadrado
    "ip"     -> producto punto negativo (inner product)

Esto permite, ante la MISMA pregunta, mostrar qué resultados/orden entrega
cada métrica y así "ver cuál de los tres métodos de distancia" se está
usando y cómo cambia el resultado.
"""
import os

import chromadb
from chromadb.config import Settings

from src.embeddings import embed_query, embed_texts
from src.ingestion import Chunk

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DISTANCES = ("cosine", "l2", "ip")


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))


def collection_name(distance: str, chunk_strategy: str | None = None) -> str:
    suffix = f"_{chunk_strategy}" if chunk_strategy else ""
    return f"rag_{distance}{suffix}"


def build_all_collections(chunks: list[Chunk]) -> None:
    """Indexa los chunks en 3 colecciones (una por métrica de distancia).
    Se conserva la estrategia de chunking como metadata para poder filtrar
    el eval por 'heading' vs 'paragraph'."""
    client = get_chroma_client()
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    for distance in DISTANCES:
        name = collection_name(distance)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": distance},
        )
        collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {
                    "source": c.source,
                    "doc_type": c.doc_type,
                    "chunk_strategy": c.chunk_strategy,
                    "heading": c.heading or "",
                }
                for c in chunks
            ],
        )


def query_all_distances(query: str, k: int = 5, chunk_strategy: str | None = None) -> dict:
    """Ejecuta la misma consulta contra las 3 colecciones y devuelve, por
    cada métrica de distancia, los chunks recuperados con su score crudo
    (tal como lo entrega Chroma para esa métrica) — así se ve explícitamente
    cuál método se usó y qué resultado produjo."""
    client = get_chroma_client()
    q_vector = embed_query(query).tolist()

    results = {}
    for distance in DISTANCES:
        collection = client.get_collection(collection_name(distance))
        where = {"chunk_strategy": chunk_strategy} if chunk_strategy else None
        res = collection.query(
            query_embeddings=[q_vector],
            n_results=k,
            where=where,
        )
        results[distance] = {
            "ids": res["ids"][0],
            "documents": res["documents"][0],
            "metadatas": res["metadatas"][0],
            "distances": res["distances"][0],  # semántica difiere según `distance`
        }
    return results
