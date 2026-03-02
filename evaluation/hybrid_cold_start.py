import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_cold_start_recommend(
    user_text,
    df,
    embeddings,
    cluster_affinity,
    top_k_docs=5,
    confidence_threshold=0.25
):

    user_embedding = _model.encode([user_text])
    similarities = cosine_similarity(user_embedding, embeddings)[0]

    df_local = df.copy()
    df_local["similarity"] = similarities

    max_conf = float(cluster_affinity.max())

    if max_conf < confidence_threshold:
        mode = "semantic-only"
        recommendations = (
            df_local
            .sort_values("similarity", ascending=False)
            .head(top_k_docs)
        )
    else:
        mode = "cluster-aware"
        top_clusters = cluster_affinity.head(3).index.tolist()
        recommendations = (
            df_local[df_local["cluster"].isin(top_clusters)]
            .sort_values("similarity", ascending=False)
            .head(top_k_docs)
        )

    return mode, recommendations
