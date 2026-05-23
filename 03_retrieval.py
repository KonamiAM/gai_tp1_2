"""
TP2 RAG — Script 3 : Query & Retrieval
Encode la question, recherche dans Qdrant, retourne les passages pertinents.
"""

import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "BAAI/bge-m3"
COLLECTION_NAME = "rag_tp2"
TOP_K           = 5             # nombre de passages récupérés
SCORE_THRESHOLD = 0.35          # filtrer les résultats trop peu pertinents

# Mode de connexion Qdrant : 'local' (Colab) ou 'cloud' (déploiement)
QDRANT_MODE     = os.getenv("QDRANT_MODE", "local")
QDRANT_URL      = os.getenv("QDRANT_URL", "https://xxxx.qdrant.io:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
LOCAL_PATH      = "./qdrant_local"

# ── Initialisation (singleton pour Streamlit) ──────────────────────────────────
_model  = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Chargement du modèle {MODEL_NAME}…")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "cloud":
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client = QdrantClient(path=LOCAL_PATH)
    return _client


# ── Embedding de la requête ────────────────────────────────────────────────────

def embed_query(question: str) -> List[float]:
    """
    Encode la question utilisateur en vecteur 1024d.
    Utilise le même modèle que l'indexation pour cohérence.
    """
    model = get_model()
    vec = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.tolist()


# ── Recherche vectorielle ──────────────────────────────────────────────────────

def retrieve(
    question: str,
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Pipeline complet de retrieval :
    1. Encode la question
    2. Recherche dans Qdrant
    3. Filtre par score
    4. Retourne les passages triés par pertinence

    Returns:
        Liste de dicts {"text", "source", "score", "chunk_idx"}
    """
    query_vector = embed_query(question)
    client       = get_qdrant_client()

    # Filtre optionnel par document source
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=score_threshold,
    )

    passages = [
        {
            "text":      r.payload["text"],
            "source":    r.payload.get("source", ""),
            "score":     round(r.score, 4),
            "chunk_idx": r.payload.get("chunk_idx", -1),
        }
        for r in results
    ]

    return passages


# ── Construction du contexte ──────────────────────────────────────────────────

def build_context(passages: List[Dict[str, Any]],
                  max_chars: int = 3000) -> str:
    """
    Concatène les passages récupérés en un bloc de contexte
    pour le prompt LLM. Limite la taille totale.
    """
    context_parts = []
    total = 0

    for i, p in enumerate(passages):
        header = f"[Source {i+1}: {p['source']} — score {p['score']}]"
        block  = f"{header}\n{p['text']}"
        if total + len(block) > max_chars:
            break
        context_parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(context_parts)


# ── Affichage debug ───────────────────────────────────────────────────────────

def display_results(passages: List[Dict[str, Any]]) -> None:
    if not passages:
        print("Aucun passage pertinent trouvé.")
        return
    print(f"\n{len(passages)} passages récupérés :\n")
    for i, p in enumerate(passages):
        print(f"  [{i+1}] Score: {p['score']}  Source: {p['source']}")
        print(f"       {p['text'][:150].strip()}…\n")


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "Comment le RAG améliore-t-il la précision d'un LLM ?"

    print(f"Question : {question}\n")
    passages = retrieve(question, top_k=3)
    display_results(passages)

    context = build_context(passages)
    print("Contexte construit :\n")
    print(context[:600])
