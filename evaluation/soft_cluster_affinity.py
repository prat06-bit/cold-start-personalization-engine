import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_soft_cluster_affinity(user_text, df, embeddings):
    """
    Returns soft probability distribution over clusters
    """

    # Embed user text
    user_emb = _model.encode([user_text])

    # Similarity to all documents
    sims = cosine_similarity(user_emb, embeddings)[0]

    df_local = df.copy()
    df_local["similarity"] = sims

    # Ignore noise
    df_local = df_local[df_local["cluster"] != -1]

    # Mean similarity per cluster
    cluster_scores = df_local.groupby("cluster")["similarity"].mean()

    # Softmax normalization
    exp_scores = np.exp(cluster_scores)
    probabilities = exp_scores / exp_scores.sum()

    return probabilities.sort_values(ascending=False)
