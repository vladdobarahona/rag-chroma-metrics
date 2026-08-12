import pandas as pd
import streamlit as st

from src.metrics import DISTANCE_FUNCS, cosine_distance, dot_product_distance, euclidean_distance, evaluate_retrieval
from src.vectorstore import query_all_distances

st.set_page_config(page_title="Dashboard de métricas", page_icon="📊", layout="wide")
st.title("📊 Comparador de métodos de distancia + calidad de recuperación")

tab1, tab2 = st.tabs(["🧭 Comparar distancias en vivo", "🎯 Precision / Recall / MRR"])

# ---------- Tab 1: comparación en vivo por query ----------
with tab1:
    st.subheader("Misma pregunta, tres métricas de distancia")
    st.caption(
        "Cada colección de Chroma fue creada con `hnsw:space` distinto "
        "(cosine / l2 / ip). Aquí se ve, para la misma pregunta, qué "
        "resultado (orden y score) entrega cada una."
    )
    query = st.text_input("Escribe una pregunta de prueba")
    k = st.slider("Top-k", 1, 10, 5, key="tab1_k")

    if query:
        with st.spinner("Consultando las 3 colecciones..."):
            results = query_all_distances(query, k=k)

        cols = st.columns(3)
        for col, distance in zip(cols, ["cosine", "l2", "ip"]):
            with col:
                st.markdown(f"### `{distance}`")
                res = results[distance]
                df = pd.DataFrame(
                    {
                        "fuente": [m["source"] for m in res["metadatas"]],
                        "chunking": [m["chunk_strategy"] for m in res["metadatas"]],
                        "distancia": [round(d, 4) for d in res["distances"]],
                    }
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.info(
            "💡 Con embeddings normalizados, `cosine` e `ip` suelen producir "
            "el **mismo orden** de resultados (la relación es monotónica). "
            "`l2` puede diferir porque penaliza también la magnitud del "
            "vector, no solo su ángulo."
        )

    st.divider()
    st.subheader("Fórmulas usadas (cálculo manual, para verificación)")
    st.code(
        """
cosine_distance(a, b) = 1 - (a·b) / (‖a‖ · ‖b‖)
euclidean_distance(a, b) = ‖a - b‖₂
dot_product_distance(a, b) = -(a·b)      # negado para que "menor = más cercano"
        """,
        language="text",
    )

# ---------- Tab 2: precision/recall/MRR contra el eval set ----------
with tab2:
    st.subheader("Evaluación de recuperación (eval/eval_dataset.json)")
    st.caption(
        "Para cada pregunta del dataset de referencia, se mide si los "
        "chunks recuperados vienen del documento esperado y contienen las "
        "palabras clave esperadas — comparado entre las 3 métricas de "
        "distancia y, opcionalmente, filtrado por estrategia de chunking."
    )

    k_eval = st.slider("k para precision@k / recall@k", 1, 10, 5, key="tab2_k")
    strategy_filter = st.selectbox(
        "Filtrar por estrategia de chunking",
        options=[None, "heading", "paragraph"],
        format_func=lambda s: {
            None: "Ambas (mixto)",
            "heading": "Solo chunking por título",
            "paragraph": "Solo chunking por párrafo",
        }[s],
        key="tab2_strategy",
    )

    if st.button("▶️ Correr evaluación"):
        with st.spinner("Evaluando cada pregunta contra las 3 métricas..."):
            df = evaluate_retrieval(query_all_distances, k=k_eval, chunk_strategy=strategy_filter)
        st.session_state["eval_df"] = df

    if "eval_df" in st.session_state:
        df = st.session_state["eval_df"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### Promedio por métrica de distancia")
        precision_col = [c for c in df.columns if c.startswith("precision")][0]
        recall_col = [c for c in df.columns if c.startswith("recall")][0]
        summary = df.groupby("distance")[[precision_col, recall_col, "mrr"]].mean().round(3)
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(summary)

        st.info(
            "💡 Corre esta evaluación filtrando primero por `heading` y "
            "luego por `paragraph` (o compara `structured` vs "
            "`unstructured` ajustando `eval_dataset.json`) para validar la "
            "hipótesis: ¿el chunking que respeta la estructura del PDF "
            "mejora el recall a igual k?"
        )
