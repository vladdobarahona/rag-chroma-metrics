"""Orquesta retrieval (con la métrica de distancia elegida) + generación."""
from src.embeddings import chat_completion
from src.vectorstore import query_all_distances

SYSTEM_PROMPT = (
    "Eres un asistente que responde ÚNICAMENTE con base en el CONTEXTO "
    "proporcionado, extraído de documentos PDF del usuario. Si el contexto "
    "no contiene la respuesta, dilo explícitamente. Cita el nombre del "
    "documento fuente entre paréntesis al final de cada afirmación."
)


def answer_question(query: str, distance: str = "cosine", k: int = 4,
                     chunk_strategy: str | None = None) -> dict:
    """Recupera top-k chunks usando la métrica de distancia indicada y
    genera la respuesta. Devuelve también el detalle de recuperación para
    poder mostrar en la UI qué método de distancia se usó y con qué scores."""
    all_results = query_all_distances(query, k=k, chunk_strategy=chunk_strategy)
    retrieval = all_results[distance]

    context_blocks = [
        f"[Fuente: {m['source']} | sección: {m.get('heading') or 'N/A'} | "
        f"distancia({distance})={dist:.4f}]\n{doc}"
        for doc, m, dist in zip(retrieval["documents"], retrieval["metadatas"], retrieval["distances"])
    ]
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(sin resultados)"

    user_prompt = f"CONTEXTO:\n{context}\n\nPREGUNTA: {query}"
    response = chat_completion(SYSTEM_PROMPT, user_prompt)

    return {
        "answer": response,
        "distance_used": distance,
        "retrieval": retrieval,
        "all_distances": all_results,
    }
