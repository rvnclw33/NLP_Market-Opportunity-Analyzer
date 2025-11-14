# src/topic_modeling.py
"""
Robust topic modeling pipeline for short texts / reviews.

Behavior:
 - Loads data from data/processed/processed_reviews.csv (expects 'text' and optionally 'title' and 'published_at')
 - Builds text_clean column (title + text)
 - Embeds with SentenceTransformer
 - Reduces with UMAP (param clipping)
 - Clusters with HDBSCAN (fallback to AgglomerativeClustering)
 - Summarizes clusters (TF-IDF top terms or token-frequency fallback)
 - Writes outputs:
     outputs/docs_with_topics.csv
     outputs/topics_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import re
import json
import sys

# -------------------- config / paths --------------------
IN_PATH = Path("data/processed/processed_reviews.csv")
OUT_TOPICS = Path("outputs/topics_summary.csv")
OUT_DOCS = Path("outputs/docs_with_topics.csv")
OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)
OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)

# Label to assign to single-document cases (choose -1 for 'noise' or 0 for a valid topic id)
SINGLE_DOC_LABEL = 0

# Defaults (tune as needed)
SBERT_MODEL = "all-MiniLM-L6-v2"
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS = 15
DEFAULT_MIN_CLUSTER_SIZE = 30
FALLBACK_MAX_CLUSTERS = 8

# -------------------- utilities --------------------
def load_data(in_path=IN_PATH):
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    try:
        df = pd.read_csv(in_path, low_memory=False, parse_dates=['published_at'])
    except Exception:
        df = pd.read_csv(in_path, low_memory=False)
    df = df.fillna("")
    if 'title' in df.columns:
        df['text_clean'] = (df['title'].astype(str).fillna('') + ". " + df.get('text', "").astype(str)).astype(str)
    else:
        df['text_clean'] = df.get('text', "").astype(str)
    return df

def top_terms_for_group(texts, topn=8):
    """
    Returns top terms for a list of texts.
    Tries TF-IDF first; if TF-IDF fails due to empty vocabulary, falls back to token frequency.
    """
    if not texts:
        return []

    # Guard: all-empty?
    if all((not t) or t.isspace() for t in texts):
        return []

    try:
        vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            raise ValueError("empty vocabulary after TF-IDF")
        sums = np.asarray(X.sum(axis=0)).ravel()
        idx = np.argsort(sums)[::-1][:topn]
        terms = [vec.get_feature_names_out()[i] for i in idx]
        return terms
    except Exception:
        # Fallback: simple unigram frequency excluding stop words & short tokens
        tokens = []
        for t in texts:
            t2 = re.sub(r"[^a-z0-9\s]", " ", str(t).lower())
            for tok in t2.split():
                if len(tok) <= 2:  # skip short tokens
                    continue
                if tok in ENGLISH_STOP_WORDS:
                    continue
                if tok.isdigit():
                    continue
                tokens.append(tok)
        if not tokens:
            return []
        counts = Counter(tokens)
        most = [w for w, _ in counts.most_common(topn)]
        return most

def summarize_clusters(df):
    groups = []
    for c in sorted(df.topic_id.unique()):
        group = df[df.topic_id == c]
        texts = group['text_clean'].tolist()
        top_terms = top_terms_for_group(texts, topn=10)
        examples = []
        if len(group) > 0:
            n_examples = min(5, len(group))
            try:
                examples = group.sample(n_examples, random_state=1)['text_clean'].tolist()
            except Exception:
                examples = group['text_clean'].tolist()[:n_examples]
        groups.append({
            'topic_id': int(c),
            'n_mentions': int(len(group)),
            'top_terms': top_terms,
            'examples': examples
        })
    return pd.DataFrame(groups)

# -------------------- embedding & clustering --------------------
def embed_and_cluster(
    docs,
    model_name=SBERT_MODEL,
    umap_n=UMAP_N_COMPONENTS,
    umap_n_neighbors=UMAP_N_NEIGHBORS,
    min_cluster_size=None,
    fallback_max_clusters=FALLBACK_MAX_CLUSTERS,
):
    """
    Embed docs with SBERT, reduce with UMAP, and cluster with HDBSCAN.
    Defensive behavior:
      - if n_docs == 0 -> return empty arrays
      - if n_docs == 1 -> assign SINGLE_DOC_LABEL and skip clustering
      - clip UMAP/HDBSCAN params relative to n_docs
      - fallback to AgglomerativeClustering if HDBSCAN fails or produces <=1 real cluster
    """
    n_docs = len(docs)
    if n_docs == 0:
        print("[info] No documents to process.")
        return np.array([], dtype=int), np.zeros((0, 0)), np.zeros((0, 0))

    model = SentenceTransformer(model_name)
    embs = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    if n_docs == 1:
        print(f"[info] Only 1 document found. Assigning topic {SINGLE_DOC_LABEL} and skipping clustering.")
        # ensure embs_reduced is 2D
        if embs.ndim == 1:
            embs_reduced = np.atleast_2d(embs)
        else:
            embs_reduced = embs
        labels = np.array([SINGLE_DOC_LABEL], dtype=int)
        return labels, embs, embs_reduced

    # for n_docs >= 2, clip params
    if min_cluster_size is None:
        min_cluster_size = max(2, n_docs // 100)
    min_cluster_size = min(min_cluster_size, max(1, n_docs - 1))

    umap_n = min(max(2, umap_n), n_docs)
    umap_n_neighbors = min(max(2, umap_n_neighbors), max(2, n_docs - 1))

    print(f"[info] n_docs={n_docs}, umap_n={umap_n}, umap_n_neighbors={umap_n_neighbors}, min_cluster_size={min_cluster_size}")

    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n, metric='cosine', random_state=42)
    embs_reduced = reducer.fit_transform(embs)

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric='euclidean')
        labels = clusterer.fit_predict(embs_reduced)
        unique_labels = set(labels.tolist())
        real_clusters = unique_labels - {-1}
        if len(real_clusters) <= 1:
            raise RuntimeError(f"HDBSCAN produced too few clusters: {unique_labels}")
        print(f"[info] HDBSCAN produced {len(real_clusters)} clusters (+ noise if any).")
    except Exception as e:
        fallback_k = min(fallback_max_clusters, max(2, n_docs // max(1, min_cluster_size)))
        fallback_k = min(fallback_k, n_docs)
        print(f"[warning] HDBSCAN failed or produced <=1 cluster. Falling back to AgglomerativeClustering (k={fallback_k}). Reason: {e}")
        agg = AgglomerativeClustering(n_clusters=max(2, fallback_k))
        labels = agg.fit_predict(embs_reduced)

    return labels, embs, embs_reduced

# -------------------- main --------------------
def main():
    print("Loading data...")
    try:
        df = load_data()
    except Exception as e:
        print("[error] Failed to load data:", e)
        sys.exit(1)

    docs = df['text_clean'].tolist()
    print("Embedding and clustering docs (SBERT -> UMAP -> HDBSCAN)...")
    safe_min_cluster = max(2, min(DEFAULT_MIN_CLUSTER_SIZE, max(2, len(df) // 100)))
    labels, embs, embs_reduced = embed_and_cluster(docs, min_cluster_size=safe_min_cluster)

    df['topic_id'] = labels

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    OUT_TOPICS.parent.mkdir(parents=True, exist_ok=True)

    print("Saving docs with topic assignments...")
    df.to_csv(OUT_DOCS, index=False)

    print("Summarizing clusters...")
    clusters_df = summarize_clusters(df)
    clusters_df = clusters_df.sort_values(['n_mentions'], ascending=False)
    clusters_df.to_csv(OUT_TOPICS, index=False)

    print("Saved topics summary to", OUT_TOPICS)
    print("Done.")

if __name__ == "__main__":
    main()
