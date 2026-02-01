# evaluation/cluster_evaluation.py

from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_clustering(embeddings, clusters):
    mask = clusters != -1  # remove noise
    filtered_embeddings = embeddings[mask]
    filtered_clusters = clusters[mask]

    if len(set(filtered_clusters)) < 2:
        return None

    return silhouette_score(filtered_embeddings, filtered_clusters)
