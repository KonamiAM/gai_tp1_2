"""
TP2 RAG — Script 2 : Qdrant embarqué → Qdrant Cloud
Phase 1 : stockage local (Colab, développement)
Phase 2 : migration vers Qdrant Cloud (déploiement)
"""

import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

# ── Configuration ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_tp2"
VECTOR_SIZE     = 1024          # dimension bge-m3
DISTANCE        = Distance.COSINE

# Qdrant Cloud — remplir avant la migration
QDRANT_URL      = os.getenv("QDRANT_URL", "https://xxxx.qdrant.io:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "votre_clé_qdrant_cloud")

# Chemin local (mode embarqué Colab)
LOCAL_PATH      = "./qdrant_local"


# ── Connexion ─────────────────────────────────────────────────────────────────

def get_local_client() -> QdrantClient:
    """Client Qdrant embarqué (stockage sur disque Colab)."""
    print(f"Connexion Qdrant LOCAL → {LOCAL_PATH}")
    return QdrantClient(path=LOCAL_PATH)


def get_cloud_client() -> QdrantClient:
    """Client Qdrant Cloud (déploiement)."""
    if "xxxx" in QDRANT_URL:
        raise ValueError(
            "Définissez QDRANT_URL et QDRANT_API_KEY "
            "(variables d'env ou directement dans le script)"
        )
    print(f"Connexion Qdrant CLOUD → {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_client(mode: str = "local") -> QdrantClient:
    """mode = 'local' | 'cloud'"""
    return get_local_client() if mode == "local" else get_cloud_client()


# ── Gestion de la collection ──────────────────────────────────────────────────

def create_collection(client: QdrantClient,
                      collection: str = COLLECTION_NAME,
                      recreate: bool = False) -> None:
    """Crée la collection si elle n'existe pas (ou la recrée)."""
    existing = [c.name for c in client.get_collections().collections]

    if collection in existing:
        if recreate:
            print(f"Suppression de la collection existante '{collection}'…")
            client.delete_collection(collection)
        else:
            print(f"Collection '{collection}' déjà présente.")
            return

    print(f"Création de la collection '{collection}'…")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=DISTANCE,
            on_disk=False,       # True si grande base (> 1 Go)
        ),
    )

    # Index sur le champ 'source' pour filtrer par document
    client.create_payload_index(
        collection_name=collection,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("  Collection et index créés.")


# ── Indexation ────────────────────────────────────────────────────────────────

def index_records(client: QdrantClient,
                  records: List[Dict[str, Any]],
                  collection: str = COLLECTION_NAME,
                  batch_size: int = 64) -> None:
    """
    Indexe une liste de records (sortie du script 01).
    Chaque record : {"text", "embedding", "source", "type", "chunk_idx"}
    """
    total = len(records)
    print(f"Indexation de {total} points dans '{collection}'…")

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=rec["embedding"],
                payload={
                    "text":      rec["text"],
                    "source":    rec["source"],
                    "type":      rec["type"],
                    "chunk_idx": rec["chunk_idx"],
                },
            )
            for rec in batch
        ]
        client.upsert(collection_name=collection, points=points)
        print(f"  Batch {i // batch_size + 1} / {-(-total // batch_size)} OK")

    info = client.get_collection(collection)
    print(f"\nIndexation terminée — {info.points_count} points dans la collection.")


# ── Recherche ─────────────────────────────────────────────────────────────────

def search(client: QdrantClient,
           query_vector: List[float],
           top_k: int = 5,
           collection: str = COLLECTION_NAME,
           source_filter: str = None) -> List[Dict[str, Any]]:
    """
    Recherche les top_k chunks les plus proches.
    Filtre optionnel par source (nom de fichier ou URL).
    """
    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )]
        )

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "text":   r.payload["text"],
            "source": r.payload["source"],
            "score":  round(r.score, 4),
            "chunk_idx": r.payload.get("chunk_idx", -1),
        }
        for r in results
    ]


# ── Migration local → Cloud ───────────────────────────────────────────────────

def migrate_to_cloud(batch_size: int = 100) -> None:
    """
    Copie tous les points du Qdrant local vers Qdrant Cloud.
    À exécuter une seule fois avant le déploiement Streamlit.
    """
    local  = get_local_client()
    cloud  = get_cloud_client()

    create_collection(cloud, recreate=True)

    offset = None
    total_migrated = 0

    print("Migration locale → Cloud…")
    while True:
        records, next_offset = local.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not records:
            break

        points = [
            PointStruct(
                id=str(r.id),
                vector=r.vector,
                payload=r.payload,
            )
            for r in records
        ]
        cloud.upsert(collection_name=COLLECTION_NAME, points=points)
        total_migrated += len(points)
        print(f"  {total_migrated} points migrés…")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nMigration terminée — {total_migrated} points dans Qdrant Cloud.")


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Simuler des records (en pratique : sortie du script 01)
    model = SentenceTransformer("BAAI/bge-m3")

    sample_records = [
        {
            "text": "Le RAG combine recherche et génération de texte.",
            "embedding": model.encode(
                "Le RAG combine recherche et génération de texte.",
                normalize_embeddings=True
            ).tolist(),
            "source": "demo.txt",
            "type": "txt",
            "chunk_idx": 0,
        },
        {
            "text": "RAG grounds LLM answers in retrieved documents.",
            "embedding": model.encode(
                "RAG grounds LLM answers in retrieved documents.",
                normalize_embeddings=True
            ).tolist(),
            "source": "demo.txt",
            "type": "txt",
            "chunk_idx": 1,
        },
    ]

    # ── Phase 1 : local ──
    client = get_client("local")
    create_collection(client, recreate=True)
    index_records(client, sample_records)

    # Test de recherche locale
    q_vec = model.encode(
        "Comment fonctionne le RAG ?",
        normalize_embeddings=True
    ).tolist()
    hits = search(client, q_vec, top_k=2)
    print("\nRésultats de recherche (local) :")
    for h in hits:
        print(f"  [{h['score']}] {h['text'][:80]}")

    # ── Phase 2 : migration Cloud (décommentez quand prêt) ──
    # migrate_to_cloud()
