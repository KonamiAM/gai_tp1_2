# TP2 RAG — Pipeline complet

Stack : BAAI/bge-m3 · Qdrant Cloud · Groq Llama 3.3 70B · Streamlit

---

## Structure des fichiers

```
rag_tp2/
├── 01_ingestion_embedding.py   # Chargement docs + chunking + embedding
├── 02_qdrant_store.py          # Qdrant local (dev) + migration Cloud
├── 03_retrieval.py             # Encode query + recherche vectorielle
├── 04_generation_groq.py       # Prompt RAG + appel Groq (stream)
├── 05_app_streamlit.py         # Interface utilisateur complète
├── requirements.txt
└── .streamlit/
    └── secrets.toml            # Clés API (ne pas committer)
```

---

## Ordre d'exécution (Google Colab)

```python
# 1. Installer les dépendances
!pip install -r requirements.txt

# 2. Indexer vos documents
# → Modifier la liste sources dans 01_ingestion_embedding.py
# → Exécuter le script, récupérer la variable `records`

# 3. Stocker dans Qdrant local
# → Appeler index_records(client, records) depuis 02_qdrant_store.py

# 4. Tester le retrieval
# → python 03_retrieval.py

# 5. Tester la génération
# → python 04_generation_groq.py

# 6. Lancer Streamlit en local
# → streamlit run 05_app_streamlit.py
```

---

## Migration vers Qdrant Cloud

1. Créer un cluster gratuit sur https://cloud.qdrant.io
2. Copier l'URL et la clé API dans `.streamlit/secrets.toml`
3. Décommenter `migrate_to_cloud()` dans `02_qdrant_store.py`
4. Exécuter la migration (une seule fois)

---

## Déploiement Streamlit Community Cloud

1. Pusher le dépôt sur GitHub (sans `secrets.toml`)
2. Aller sur https://share.streamlit.io → New app
3. Sélectionner le repo + `05_app_streamlit.py`
4. Dans Settings → Secrets, coller le contenu de `secrets.toml`
5. Deploy !

---

## Variables d'environnement requises

| Variable        | Description                     |
|----------------|---------------------------------|
| GROQ_API_KEY   | Clé API Groq (gratuite)         |
| QDRANT_URL     | URL du cluster Qdrant Cloud     |
| QDRANT_API_KEY | Clé API Qdrant Cloud            |
| QDRANT_MODE    | `cloud` ou `local`              |

---

## Obtenir les clés gratuites

- **Groq** : https://console.groq.com → API Keys
- **Qdrant Cloud** : https://cloud.qdrant.io → Cluster gratuit 1 GB
