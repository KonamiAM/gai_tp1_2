"""
TP2 RAG — Script 4 : Génération — Groq API + Llama 3.3 70B
Construit le prompt RAG, appelle Groq, retourne la réponse.
"""

import os
from typing import List, Dict, Any, Generator

from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "votre_clé_groq_ici")
GROQ_MODEL    = "llama-3.3-70b-versatile"  # modèle gratuit Groq
MAX_TOKENS    = 1024
TEMPERATURE   = 0.2   # faible pour des réponses factuelles ancrées

# Prompt système multilingue (détecte la langue de la question)
SYSTEM_PROMPT = """Tu es un assistant expert qui répond aux questions en se basant UNIQUEMENT sur le contexte fourni.

Règles :
- Réponds dans la même langue que la question (français, anglais ou arabe).
- Si la réponse se trouve dans le contexte, donne une réponse précise et structurée.
- Si le contexte ne contient pas la réponse, dis-le clairement : "Je ne trouve pas cette information dans les documents fournis."
- Ne fabrique pas d'informations. Ne complète pas avec des connaissances générales.
- Cite les sources entre crochets [Source N] quand tu utilises un passage spécifique.
- Sois concis et factuel."""

# ── Client Groq (singleton) ───────────────────────────────────────────────────
_groq_client = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if "votre_clé" in GROQ_API_KEY:
            raise ValueError(
                "Définissez votre clé Groq : "
                "export GROQ_API_KEY='gsk_...' "
                "ou modifiez GROQ_API_KEY dans ce script."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Construction du prompt ────────────────────────────────────────────────────

def build_rag_prompt(question: str,
                     context: str) -> List[Dict[str, str]]:
    """
    Construit les messages pour l'API Groq.
    Format : [system, user]
    """
    user_content = f"""Contexte extrait des documents :

{context}

---

Question : {question}

Réponds en te basant uniquement sur le contexte ci-dessus."""

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


# ── Génération standard ───────────────────────────────────────────────────────

def generate(question: str,
             context: str,
             max_tokens: int = MAX_TOKENS,
             temperature: float = TEMPERATURE) -> Dict[str, Any]:
    """
    Appelle Groq et retourne la réponse complète.

    Returns:
        {
          "answer":     str,   # texte de la réponse
          "model":      str,   # modèle utilisé
          "input_tokens": int,
          "output_tokens": int,
        }
    """
    client   = get_groq_client()
    messages = build_rag_prompt(question, context)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    choice = response.choices[0]
    usage  = response.usage

    return {
        "answer":        choice.message.content,
        "model":         response.model,
        "input_tokens":  usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


# ── Génération en streaming (pour Streamlit) ──────────────────────────────────

def generate_stream(question: str,
                    context: str,
                    max_tokens: int = MAX_TOKENS,
                    temperature: float = TEMPERATURE) -> Generator[str, None, None]:
    """
    Version streaming de generate().
    Utilisée dans Streamlit avec st.write_stream().

    Yields:
        Morceaux de texte au fur et à mesure de la génération.
    """
    client   = get_groq_client()
    messages = build_rag_prompt(question, context)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ── Pipeline RAG complet (retrieval + génération) ─────────────────────────────

def rag_answer(question: str,
               top_k: int = 5,
               stream: bool = False) -> Any:
    """
    Pipeline RAG end-to-end :
    Retrieval (script 03) → build_context → Groq

    Args:
        question: question de l'utilisateur
        top_k:    nombre de passages à récupérer
        stream:   si True, retourne un générateur (pour Streamlit)

    Returns:
        Si stream=False : dict {"answer", "model", "passages", ...}
        Si stream=True  : (generator, passages)
    """
    # Import ici pour éviter les imports circulaires
    from retrieval import retrieve, build_context  # noqa

    passages = retrieve(question, top_k=top_k)
    context  = build_context(passages)

    if not passages:
        empty = "Je ne trouve pas cette information dans les documents fournis."
        if stream:
            return (x for x in [empty]), []
        return {"answer": empty, "passages": [], "model": GROQ_MODEL,
                "input_tokens": 0, "output_tokens": 0}

    if stream:
        return generate_stream(question, context), passages

    result   = generate(question, context)
    result["passages"] = passages
    return result


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test direct sans retrieval (contexte fictif)
    test_context = """[Source 1: rag_intro.txt — score 0.92]
Le Retrieval-Augmented Generation (RAG) est une architecture qui améliore 
les LLMs en leur fournissant des documents pertinents au moment de la requête.
Cela réduit les hallucinations et ancre les réponses dans des faits vérifiables.

---

[Source 2: rag_intro.txt — score 0.87]
RAG combines a retrieval system (often vector search) with a language model.
The retriever finds relevant passages; the LLM synthesizes them into an answer."""

    question = "Quels sont les avantages du RAG ?"
    print(f"Question : {question}\n")

    # Mode normal
    result = generate(question, test_context)
    print(f"Réponse ({result['model']}) :\n")
    print(result["answer"])
    print(f"\nTokens — entrée: {result['input_tokens']}, "
          f"sortie: {result['output_tokens']}")

    # Mode streaming
    print("\n--- Streaming ---")
    for token in generate_stream(question, test_context):
        print(token, end="", flush=True)
    print()
