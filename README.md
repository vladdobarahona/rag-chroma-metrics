# Ejercicio teórico-práctico de RAG: distancias vectoriales + calidad de recuperación

Laboratorio para entender, de forma medible, cómo la **estructura de un PDF**
(con títulos/jerarquía vs. texto plano sin estructura) y la **métrica de
distancia** usada en la base vectorial (coseno, euclídea/L2, producto punto)
afectan la calidad de un pipeline RAG.

## Objetivo pedagógico

1. Cargar documentos PDF **con estructura** (títulos, subtítulos, secciones)
   y **sin estructura** (texto corrido, sin jerarquía) al mismo repositorio.
2. Trocear (chunking) cada tipo de documento con dos estrategias distintas:
   - `chunking por título/sección` (usa la jerarquía del PDF estructurado)
   - `chunking por párrafo/tamaño fijo` (única opción viable en el PDF sin
     estructura)
3. Indexar los mismos chunks en **tres colecciones Chroma**, cada una
   configurada con una métrica de distancia distinta (`cosine`, `l2`, `ip`).
4. Ante una misma consulta, mostrar **qué método de distancia devolvió cada
   resultado**, sus scores, y compararlos.
5. Medir la calidad real de la recuperación con **precision@k / recall@k /
   MRR** contra un dataset de preguntas con respuestas esperadas
   (`eval/eval_dataset.json`), para cada métrica de distancia y para cada
   estrategia de chunking. Esto responde la pregunta de fondo: *¿el vector
   diferencia mejor cuando el chunk respeta la estructura del documento?*
6. Servir todo como un chatbot RAG en Streamlit, desplegable directo desde
   GitHub.

## Por qué estas piezas

| Pieza | Elección | Motivo |
|---|---|---|
| Vector DB | **Chroma** (embebido, `PersistentClient`) | Open source, sin servidor externo, corre dentro del contenedor de Streamlit Cloud, y permite fijar la métrica de distancia por colección vía `hnsw:space`. |
| Embeddings + LLM | **GitHub Models** (`https://models.inference.ai.azure.com`, API compatible con OpenAI) | Es el mismo catálogo que usa GitHub Copilot; se autentica con un `GITHUB_TOKEN` (gratis con cuota), sin tarjeta de crédito, ideal para un ejercicio académico que además se puede desplegar. |
| Framework RAG | Cliente `openai` puro + Chroma directo | Se evita una capa extra (LangChain/LlamaIndex) para que las 3 métricas de distancia y el cálculo de precision/recall queden explícitos y auditables en el código, que es justamente el punto pedagógico. |
| Front | **Streamlit**, multipágina | `app.py` = chatbot, `pages/1_📊_Metricas.py` = dashboard comparativo de las 3 distancias + precision/recall. |

## Estructura del repositorio

```
rag-chroma-metrics/
├── app.py                      # Chatbot RAG (Streamlit)
├── pages/
│   └── 1_📊_Metricas.py        # Dashboard: distancias + precision/recall
├── src/
│   ├── ingestion.py            # Carga PDFs, detecta estructura, chunking
│   ├── embeddings.py           # Wrapper de embeddings (GitHub Models)
│   ├── vectorstore.py          # 3 colecciones Chroma (cosine/l2/ip)
│   ├── metrics.py              # Distancias manuales + precision/recall/MRR
│   └── rag_chain.py            # Retrieval + generación de respuesta
├── documents/
│   ├── structured/             # PDFs con títulos/jerarquía (ver README ahí)
│   └── unstructured/           # PDFs de texto plano, sin jerarquía
├── eval/
│   └── eval_dataset.json       # preguntas -> chunks relevantes esperados
├── chroma_db/                  # (se genera al indexar, no versionar)
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # pega tu GITHUB_TOKEN (Settings > Developer settings > Tokens)

# 1) Coloca tus PDFs en documents/structured y documents/unstructured
# 2) Indexa (crea las 3 colecciones Chroma)
python -m src.ingestion

# 3) Levanta el chatbot
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repo a GitHub.
2. En https://share.streamlit.io conecta el repo, archivo principal `app.py`.
3. En "Secrets" del deployment agrega:
   ```toml
   GITHUB_TOKEN = "ghp_xxx..."
   ```
4. Sube `chroma_db/` ya indexado al repo (o corre un botón de
   "Reindexar" dentro de la app — incluido en `app.py`) porque Streamlit
   Cloud no persiste disco entre reinicios de contenedor.

## Qué observar durante el ejercicio

- En el dashboard de métricas, comparar cómo cambian **el orden de los
  resultados y sus scores** entre `cosine`, `l2` e `ip` para la misma
  pregunta — con embeddings normalizados, coseno e IP deberían coincidir en
  el ranking; L2 puede diferir si las magnitudes varían.
- Comparar `precision@k`/`recall@k` de las preguntas cuya respuesta vive en
  un PDF **estructurado** (chunking por título) vs. una cuyo PDF es
  **plano** (chunking por tamaño fijo) — la hipótesis a validar es que el
  chunking que respeta títulos produce chunks más semánticamente puros y
  por tanto mejor recall a igual k.
