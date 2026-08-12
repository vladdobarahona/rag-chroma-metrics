import streamlit as st

from src.ingestion import load_and_chunk_all
from src.rag_chain import answer_question
from src.vectorstore import build_all_collections

st.set_page_config(page_title="RAG · Chatbot", page_icon="💬", layout="wide")

st.title("💬 Chatbot RAG — comparador de métricas de distancia")
st.caption(
    "Chroma (open source) + embeddings/LLM de GitHub Models. "
    "Elige qué métrica de distancia usar para la recuperación y observa "
    "la fuente y el score de cada chunk recuperado."
)

with st.sidebar:
    st.header("⚙️ Configuración de recuperación")
    distance = st.selectbox(
        "Métrica de distancia",
        options=["cosine", "l2", "ip"],
        format_func=lambda d: {
            "cosine": "Coseno (similitud angular)",
            "l2": "Euclídea (L2)",
            "ip": "Producto punto (IP)",
        }[d],
    )
    chunk_strategy = st.selectbox(
        "Estrategia de chunking",
        options=[None, "heading", "paragraph"],
        format_func=lambda s: {
            None: "Ambas (mixto)",
            "heading": "Por título/sección (solo PDFs estructurados)",
            "paragraph": "Por párrafo/tamaño fijo",
        }[s],
    )
    k = st.slider("Top-k chunks a recuperar", 1, 10, 4)

    st.divider()
    st.subheader("📁 Repositorio de documentos")
    st.write("Coloca PDFs en `documents/structured/` y `documents/unstructured/`.")
    if st.button("🔄 Reindexar documentos", use_container_width=True):
        with st.spinner("Troceando PDFs y generando embeddings..."):
            chunks = load_and_chunk_all()
            build_all_collections(chunks)
        st.success(f"Reindexado: {len(chunks)} chunks en 3 colecciones (cosine/l2/ip).")

    st.divider()
    st.page_link("pages/1_📊_Metricas.py", label="Ver dashboard de métricas", icon="📊")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Pregunta algo sobre tus documentos...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Recuperando con distancia '{distance}' y generando respuesta..."):
                result = answer_question(prompt, distance=distance, k=k, chunk_strategy=chunk_strategy)
            st.markdown(result["answer"])

            with st.expander(f"🔍 Chunks recuperados (métrica: {distance})"):
                for doc, m, dist in zip(
                    result["retrieval"]["documents"],
                    result["retrieval"]["metadatas"],
                    result["retrieval"]["distances"],
                ):
                    st.markdown(
                        f"**{m['source']}** · sección: *{m.get('heading') or 'N/A'}* · "
                        f"chunking: `{m['chunk_strategy']}` · distancia (`{distance}`) = `{dist:.4f}`"
                    )
                    st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                    st.divider()

            st.session_state.history.append({"role": "assistant", "content": result["answer"]})
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(
                "No fue posible completar la consulta. Verifica que hayas "
                f"indexado documentos y configurado GITHUB_TOKEN. Detalle: {e}"
            )
