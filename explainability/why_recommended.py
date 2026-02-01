def explain_recommendation(user_text, recommendations, cluster_probs):
    """
    Explains why documents were recommended
    """

    explanations = []

    for _, row in recommendations.iterrows():
        cluster_id = row["cluster"]
        cluster_prob = cluster_probs.get(cluster_id, 0.0)

        explanation = {
            "cluster": cluster_id,
            "similarity": round(row["similarity"], 4),
            "cluster_affinity": round(cluster_prob, 4),
            "reason": (
                f"This document was recommended because it belongs to "
                f"cluster {cluster_id}, which strongly aligns with the user's intent "
                f"(cluster affinity = {round(cluster_prob, 2)}), and it has high "
                f"semantic similarity to the user query."
            )
        }

        explanations.append(explanation)

    return explanations
