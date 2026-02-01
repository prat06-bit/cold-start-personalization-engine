def show_cluster_examples(df, cluster_id, n=3):
    print(f"\n--- Cluster {cluster_id} examples ---")
    samples = df[df["cluster"] == cluster_id]["clean_text"].head(n)

    for i, text in enumerate(samples):
        print(f"\nExample {i+1}:")
        print(text[:300])
