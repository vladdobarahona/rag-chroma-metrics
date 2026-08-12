"""
Ingesta de PDFs "con estructura" y "sin estructura" al repositorio de
documentos, con dos estrategias de chunking:

  - chunk_by_heading(): usa el tamaño/negrita de fuente (vía PyMuPDF) para
    detectar títulos y agrupa el texto bajo cada título como un chunk.
    Pensado para PDFs con jerarquía real (informes, papers, manuales).

  - chunk_by_paragraph(): fallback de tamaño fijo con solapamiento, basado
    en párrafos. Es la única opción razonable para PDFs de texto plano
    (escaneados-a-texto, transcripciones, contratos sin secciones).

Ambas estrategias se aplican a *todos* los documentos para poder comparar
en el eval qué tanto ayuda (o no) respetar la estructura real del PDF.
"""
import glob
import os
import uuid
from dataclasses import dataclass, field

import fitz  # PyMuPDF

STRUCTURED_DIR = "documents/structured"
UNSTRUCTURED_DIR = "documents/unstructured"


@dataclass
class Chunk:
    id: str
    text: str
    source: str          # nombre de archivo
    doc_type: str         # "structured" | "unstructured"
    chunk_strategy: str   # "heading" | "paragraph"
    heading: str | None = None
    metadata: dict = field(default_factory=dict)


def _extract_spans(pdf_path: str) -> list[dict]:
    """Extrae texto por línea con su tamaño de fuente y si está en negrita."""
    doc = fitz.open(pdf_path)
    spans = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text:
                    continue
                sizes = [s["size"] for s in line["spans"]]
                is_bold = any("Bold" in s.get("font", "") for s in line["spans"])
                spans.append(
                    {
                        "text": line_text,
                        "size": max(sizes),
                        "bold": is_bold,
                        "page": page_num,
                    }
                )
    doc.close()
    return spans


def chunk_by_heading(pdf_path: str, doc_type: str) -> list[Chunk]:
    """Agrupa texto bajo cada título detectado por tamaño de fuente/negrita."""
    spans = _extract_spans(pdf_path)
    if not spans:
        return []

    body_size = sorted(s["size"] for s in spans)[len(spans) // 2]  # mediana
    heading_threshold = body_size * 1.15

    chunks, current_heading, buffer = [], "Introducción", []

    def flush():
        text = " ".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    source=os.path.basename(pdf_path),
                    doc_type=doc_type,
                    chunk_strategy="heading",
                    heading=current_heading,
                )
            )

    for s in spans:
        looks_like_heading = s["size"] >= heading_threshold or (s["bold"] and len(s["text"]) < 90)
        if looks_like_heading:
            flush()
            current_heading = s["text"]
            buffer = []
        else:
            buffer.append(s["text"])
    flush()
    return chunks


def chunk_by_paragraph(pdf_path: str, doc_type: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunking por caracteres con solapamiento, sobre el texto plano del PDF."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    joined = "\n\n".join(paragraphs)

    chunks = []
    start = 0
    while start < len(joined):
        end = start + size
        piece = joined[start:end]
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=piece,
                source=os.path.basename(pdf_path),
                doc_type=doc_type,
                chunk_strategy="paragraph",
                heading=None,
            )
        )
        start += size - overlap
    return chunks


def load_and_chunk_all() -> list[Chunk]:
    """Recorre documents/structured y documents/unstructured, aplica AMBAS
    estrategias de chunking a cada PDF (para poder comparar en el eval)."""
    all_chunks: list[Chunk] = []
    for doc_type, folder in (("structured", STRUCTURED_DIR), ("unstructured", UNSTRUCTURED_DIR)):
        for pdf_path in glob.glob(os.path.join(folder, "*.pdf")):
            all_chunks.extend(chunk_by_heading(pdf_path, doc_type))
            all_chunks.extend(chunk_by_paragraph(pdf_path, doc_type))
    return all_chunks


if __name__ == "__main__":
    from src.vectorstore import build_all_collections

    chunks = load_and_chunk_all()
    print(f"Chunks generados: {len(chunks)}")
    for dt in ("structured", "unstructured"):
        for strat in ("heading", "paragraph"):
            n = sum(1 for c in chunks if c.doc_type == dt and c.chunk_strategy == strat)
            print(f"  {dt:12s} / {strat:9s}: {n} chunks")

    build_all_collections(chunks)
    print("Indexado en Chroma (3 colecciones: cosine, l2, ip) ✅")
