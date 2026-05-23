"""
TP2 RAG — Script 5 : Application Streamlit
Interface utilisateur complète — déploiement via Streamlit Community Cloud
"""

import os
import time
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq

# ────────────────────────────────────────────────────────────────────────────
# Configuration (depuis st.secrets ou variables d'environnement)
# ────────────────────────────────────────────────────────────────────────────
def cfg(key: str, default: str = "") -> str:
    """Lit depuis st.secrets si disponible, sinon os.environ."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

GROQ_API_KEY    = cfg("GROQ_API_KEY")
QDRANT_URL      = cfg("QDRANT_URL")
QDRANT_API_KEY  = cfg("QDRANT_API_KEY")
QDRANT_MODE     = cfg("QDRANT_MODE", "cloud")   # 'local' ou 'cloud'
LOCAL_PATH      = "./qdrant_local"

COLLECTION_NAME = "rag_tp2"
MODEL_NAME      = "BAAI/bge-m3"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 5
SCORE_THRESHOLD = 0.35
MAX_TOKENS      = 1024
TEMPERATURE     = 0.2

SYSTEM_PROMPT = """Tu es un assistant expert qui répond aux questions en se basant UNIQUEMENT sur le contexte fourni.
Règles :
- Réponds dans la même langue que la question (français, anglais ou arabe).
- Si la réponse est dans le contexte, donne une réponse précise et structurée.
- Sinon, dis : "Je ne trouve pas cette information dans les documents fournis."
- Ne fabrique pas d'informations. Cite les sources [Source N] quand pertinent.
- Sois concis et factuel."""


# ────────────────────────────────────────────────────────────────────────────
# Singletons cachés (chargement unique grâce à @st.cache_resource)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Connexion à Qdrant…")
def load_qdrant() -> QdrantClient:
    if QDRANT_MODE == "cloud":
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=LOCAL_PATH)


@st.cache_resource(show_spinner=False)
def load_groq() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


# ────────────────────────────────────────────────────────────────────────────
# Fonctions RAG
# ────────────────────────────────────────────────────────────────────────────
def retrieve(question: str, source_filter: str = None):
    model  = load_model()
    client = load_qdrant()

    vec = model.encode(
        question, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    query_filter = None
    if source_filter and source_filter != "Tous":
        query_filter = Filter(
            must=[FieldCondition(key="source",
                                 match=MatchValue(value=source_filter))]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=TOP_K,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    )

    return [
        {
            "text":   r.payload["text"],
            "source": r.payload.get("source", ""),
            "score":  round(r.score, 4),
        }
        for r in results
    ]


def build_context(passages: list, max_chars: int = 3000) -> str:
    parts, total = [], 0
    for i, p in enumerate(passages):
        block = f"[Source {i+1}: {p['source']} — score {p['score']}]\n{p['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def stream_answer(question: str, context: str):
    groq = load_groq()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Contexte :\n\n{context}\n\n---\n\nQuestion : {question}"},
    ]
    stream = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def get_sources() -> list:
    """Récupère la liste des sources indexées dans Qdrant."""
    try:
        client  = load_qdrant()
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )[0]
        sources = sorted(set(r.payload.get("source", "") for r in results))
        return ["Tous"] + sources
    except Exception:
        return ["Tous"]


# ────────────────────────────────────────────────────────────────────────────
# Interface Streamlit
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG TP2",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")

    top_k = st.slider("Passages récupérés (top-k)", 1, 10, TOP_K)
    score_thr = st.slider("Seuil de pertinence", 0.0, 1.0, SCORE_THRESHOLD, 0.05)
    temperature = st.slider("Température LLM", 0.0, 1.0, TEMPERATURE, 0.05)

    st.markdown("---")
    st.subheader("📂 Filtrer par document")
    sources = get_sources()
    selected_source = st.selectbox("Source", sources)

    st.markdown("---")
    st.subheader("ℹ️ Stack technique")
    st.markdown("""
- **Embedding** : `BAAI/bge-m3`
- **Dimensions** : 1 024
- **Langues** : FR · EN · AR
- **Base vectorielle** : Qdrant Cloud
- **LLM** : Llama 3.3 70B (Groq)
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Assistant RAG — TP2")
st.caption("Posez une question en français, anglais ou arabe sur vos documents indexés.")
st.markdown("---")

# ── Historique de conversation ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "passages" in msg:
            with st.expander(f"📄 {len(msg['passages'])} passages récupérés"):
                for i, p in enumerate(msg["passages"]):
                    st.markdown(
                        f"**Source {i+1}** — `{p['source']}` — score `{p['score']}`"
                    )
                    st.text(p["text"][:400] + ("…" if len(p["text"]) > 400 else ""))
                    st.markdown("---")

# ── Input utilisateur ─────────────────────────────────────────────────────────
if question := st.chat_input("Votre question…"):

    # Affiche la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Réponse RAG
    with st.chat_message("assistant"):
        t0 = time.time()

        # Retrieval
        with st.spinner("Recherche dans les documents…"):
            passages = retrieve(question, selected_source)

        if not passages:
            answer = "Je ne trouve pas cette information dans les documents fournis."
            st.warning(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "passages": []}
            )
        else:
            context = build_context(passages, max_chars=3000)

            # Génération en streaming
            answer_placeholder = st.empty()
            full_answer = ""

            for token in stream_answer(question, context):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            elapsed = round(time.time() - t0, 1)
            st.caption(f"⏱ {elapsed}s · {len(passages)} passages · {GROQ_MODEL}")

            # Sources récupérées (pliables)
            with st.expander(f"📄 {len(passages)} passages récupérés"):
                for i, p in enumerate(passages):
                    st.markdown(
                        f"**Source {i+1}** — `{p['source']}` — score `{p['score']}`"
                    )
                    st.text(p["text"][:400] + ("…" if len(p["text"]) > 400 else ""))
                    st.markdown("---")

            st.session_state.messages.append({
                "role":     "assistant",
                "content":  full_answer,
                "passages": passages,
            })

# ── Bouton reset ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Effacer la conversation", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
