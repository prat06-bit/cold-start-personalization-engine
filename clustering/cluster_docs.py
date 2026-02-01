from sklearn.cluster import OPTICS
import numpy as np

def cluster_embeddings(embeddings):
    print(">> OPTICS started")

    model = OPTICS(
    metric="cosine",
    min_samples=8,
    min_cluster_size=15,
    xi=0.05,
    cluster_method="xi",
    n_jobs=-1
)

    clusters = model.fit_predict(embeddings)

    print(">> OPTICS finished")
    return clusters
