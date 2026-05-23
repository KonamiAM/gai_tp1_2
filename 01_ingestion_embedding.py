"""
TP2 RAG — Script 1 : Ingestion & Embedding
Modèle : BAAI/bge-m3 (1024 dim, multilingue FR/EN/AR)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import numpy as np

# ── Optionnel : lecture PDF et pages web ──────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SUPPORT = True
except ImportError:
    WEB_SUPPORT = False

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-m3"
CHUNK_SIZE   = 512   # caractères par chunk
CHUNK_OVERLAP = 64   # chevauchement entre chunks
BATCH_SIZE   = 32    # chunks encodés à la fois (mémoire GPU Colab)

# ── Chargement du modèle ──────────────────────────────────────────────────────
print(f"Chargement du modèle {MODEL_NAME}…")
model = SentenceTransformer(MODEL_NAME)
print(f"  Dimension de sortie : {model.get_sentence_embedding_dimension()}")


# ── Lecture de documents ──────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    """Lit un fichier texte brut."""
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str) -> str:
    """Extrait le texte d'un PDF via PyMuPDF."""
    if not PDF_SUPPORT:
        raise ImportError("Installez PyMuPDF : pip install pymupdf")
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def load_url(url: str) -> str:
    """Extrait le texte principal d'une page web."""
    if not WEB_SUPPORT:
        raise ImportError("Installez requests et beautifulsoup4")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def load_document(source: str) -> Dict[str, Any]:
    """
    Charge n'importe quelle source (fichier ou URL).
    Retourne {"text": ..., "source": ..., "type": ...}
    """
    if source.startswith("http"):
        text = load_url(source)
        return {"text": text, "source": source, "type": "web"}
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {source}")
    if path.suffix.lower() == ".pdf":
        text = load_pdf(source)
        return {"text": text, "source": source, "type": "pdf"}
    text = load_txt(source)
    return {"text": text, "source": source, "type": "txt"}


# ── Chunking ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Nettoyage basique : espaces multiples, lignes vides."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Découpe le texte en chunks de taille fixe avec chevauchement.
    Respecte les fins de phrases pour ne pas couper au milieu d'un mot.
    """
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Recule jusqu'à un espace pour ne pas couper un mot
            while end > start and text[end] not in " \n":
                end -= 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: List[str],
                 batch_size: int = BATCH_SIZE,
                 show_progress: bool = True) -> np.ndarray:
    """
    Encode une liste de chunks en vecteurs float32 (1024d).
    Retourne un tableau numpy de shape (n_chunks, 1024).
    """
    print(f"Encodage de {len(chunks)} chunks (batch={batch_size})…")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # nécessaire pour similarité cosinus
        convert_to_numpy=True,
    )
    print(f"  Shape des embeddings : {embeddings.shape}")
    return embeddings.astype("float32")


# ── Pipeline complet ──────────────────────────────────────────────────────────

def ingest(sources: List[str]) -> List[Dict[str, Any]]:
    """
    Pipeline complet pour une liste de sources.
    Retourne une liste de dicts prêts pour Qdrant :
      {
        "text":      str,       # contenu du chunk
        "embedding": list,      # vecteur 1024d
        "source":    str,       # fichier ou URL d'origine
        "type":      str,       # pdf | txt | web
        "chunk_idx": int,       # index du chunk dans le document
      }
    """
    all_records = []

    for source in sources:
        print(f"\nTraitement : {source}")
        doc = load_document(source)
        chunks = chunk_text(doc["text"])
        print(f"  {len(chunks)} chunks générés")

        embeddings = embed_chunks(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            all_records.append({
                "text":      chunk,
                "embedding": vec.tolist(),
                "source":    doc["source"],
                "type":      doc["type"],
                "chunk_idx": i,
            })

    print(f"\nTotal : {len(all_records)} chunks prêts pour l'indexation.")
    return all_records


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Remplacez par vos propres sources
    sample_sources = [
        "data/document1.txt",
        "data/rapport.pdf",
        # "https://fr.wikipedia.org/wiki/Retrieval-augmented_generation",
    ]

    # Créer un fichier de test si nécessaire
    os.makedirs("data", exist_ok=True)
    if not Path("data/document1.txt").exists():
        Path("data/document1.txt").write_text(
            "Le Retrieval-Augmented Generation (RAG) est une technique qui combine "
            "la recherche d'information et la génération de texte par un LLM. "
            "Elle permet d'ancrer les réponses dans des documents réels, "
            "réduisant ainsi les hallucinations.\n\n"
            "RAG combines information retrieval with language model generation. "
            "It grounds responses in retrieved documents, improving factual accuracy.",
            encoding="utf-8"
        )
        sample_sources = ["data/document1.txt"]

    records = ingest(sample_sources)
    print(f"\nPremier chunk : {records[0]['text'][:120]}…")
    print(f"Dimension vecteur : {len(records[0]['embedding'])}")
