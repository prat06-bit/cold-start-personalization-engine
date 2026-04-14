from data.load_data import load_and_clean_data
from embeddings.embed_text import generate_embeddings
from clustering.cluster_docs import cluster_embeddings
from explainability.inspect_clusters import show_cluster_examples
from evaluation.cold_start_personalization import cold_start_recommend
from evaluation.soft_cluster_affinity import compute_soft_cluster_affinity
from evaluation.hybrid_cold_start import hybrid_cold_start_recommend
from evaluation.cluster_evaluation import evaluate_clustering
from explainability.why_recommended import explain_recommendation


import numpy as np
import os
import umap
import pandas as pd

print("\n[1] Loading and cleaning data.")
df = load_and_clean_data("newsgroups.json")
print("Dataset shape:", df.shape)

print("\n[2] Generating sentence embeddings")
EMB_PATH = "embeddings.npy"

if os.path.exists(EMB_PATH):
    print("Loading cached embeddings from disk")
    embeddings = np.load(EMB_PATH)
else:
    print("No cache found. Generating embeddings")
    embeddings = generate_embeddings(df["clean_text"].tolist())
    np.save(EMB_PATH, embeddings)
    print("Embeddings cached to disk.")

print("Embeddings shape:", embeddings.shape)

print("\n[3] Performing density-based clustering (OPTICS)...")
clusters = cluster_embeddings(embeddings)
df["cluster"] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
noise_ratio = np.mean(clusters == -1)
coverage = 1 - noise_ratio
print("Cluster coverage:", round(coverage, 3))
print("Clusters discovered:", n_clusters)
print("Noise ratio:", round(noise_ratio, 3))


print("\n[4] Evaluating clustering quality...")
score = evaluate_clustering(embeddings, clusters)

if score is not None:
    print("Silhouette Score:", round(score, 4))

# Cluster Size Analysis
print("\n[5] Cluster size distribution (top 10):")
print(df["cluster"].value_counts().head(10))

#  Explainability
print("\n[6] Inspecting discovered clusters...")
valid_clusters = [c for c in sorted(df["cluster"].unique()) if c != -1]
for cid in valid_clusters[:2]:
    show_cluster_examples(df, cluster_id=cid, n=3)


print("\n[7] Running UMAP dimensionality reduction...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings)
df["umap_x"] = embedding_2d[:, 0]
df["umap_y"] = embedding_2d[:, 1]
print("UMAP projection complete.")
print(df[["umap_x", "umap_y", "cluster"]].head())
print("\nPipeline completed successfully.")


print("\n[8] Cold-start personalization (hybrid mode)")
user_query = "I want to learn machine learning and artificial intelligence"

# compute soft cluster affinity (you already do this)
probs = compute_soft_cluster_affinity(
    user_text=user_query,
    df=df,
    embeddings=embeddings
)
mode, recs = hybrid_cold_start_recommend(
    user_text=user_query,
    df=df,
    embeddings=embeddings,
    cluster_affinity=probs,
    top_k_docs=5
)
print(f"\nCold-start mode: {mode}")
print("\nRecommended documents:")
for _, row in recs.iterrows():
    print("\n--- Recommendation ---")
    print("Cluster:", row.get("cluster", "N/A"))
    print("Similarity:", round(row["similarity"], 4))
    print(row["text"][:400])


# Soft Cluster Probabilities

print("\n[9] Soft cluster affinity")

user_query = "I want to learn machine learning and artificial intelligence"

probs = compute_soft_cluster_affinity(
    user_text=user_query,
    df=df,
    embeddings=embeddings
)

print("\nSoft cluster distribution:")
print(probs.head(5))


print("\n[10] Explainability: Why these recommendations?")
explanations = explain_recommendation(
    user_text=user_query,
    recommendations=recs,
    cluster_probs=probs
)
for i, exp in enumerate(explanations, 1):
    print(f"\nExplanation {i}:")
    print("Cluster:", exp["cluster"])
    print("Similarity:", exp["similarity"])
    print("Cluster Affinity:", exp["cluster_affinity"])
    print("Reason:", exp["reason"])

