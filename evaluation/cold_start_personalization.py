import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the SAME embedding model
_model = SentenceTransformer("all-MiniLM-L6-v2")


def cold_start_recommend(
    user_text,
    df,
    embeddings,
    top_k_clusters=3,
    top_k_docs=5
):
    """
    Cold-start personalization:
    - user_text: new user sentence
    - df: dataframe with cluster labels
    - embeddings: document embeddings
    """

    # 1. Embed user text
    user_embedding = _model.encode([user_text])

    # 2. Compute similarity with all documents
    similarities = cosine_similarity(user_embedding, embeddings)[0]

    # 3. Attach similarity to dataframe
    df_local = df.copy()
    df_local["similarity"] = similarities

    # 4. Ignore noise cluster (-1)
    df_local = df_local[df_local["cluster"] != -1]

    # 5. Aggregate similarity by cluster
    cluster_scores = (
        df_local
        .groupby("cluster")["similarity"]
        .mean()
        .sort_values(ascending=False)
    )

    top_clusters = cluster_scores.head(top_k_clusters).index.tolist()

    # 6. Recommend top documents from those clusters
    recommendations = (
        df_local[df_local["cluster"].isin(top_clusters)]
        .sort_values("similarity", ascending=False)
        .head(top_k_docs)
    )

    return top_clusters, recommendations
