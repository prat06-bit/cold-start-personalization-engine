STOPWORDS = {
    "i","the","is","and","to","of","in","a","for","on","with","that","this","it","as"
}

def extract_keywords(text, top_k=5):
    words = text.lower().split()
    words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
    return list(set(words))[:top_k]


def explain_recommendation(user_text, recommendations, cluster_probs, threshold=0.25):

    explanations = []
    user_keywords = extract_keywords(user_text)

    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):

        cluster_id = row["cluster"]
        similarity = round(row["similarity"], 4)
        cluster_prob = round(cluster_probs.get(cluster_id, 0.0), 4)

        doc_text = row.get("text", "")
        doc_keywords = extract_keywords(doc_text)

        matched_keywords = list(set(user_keywords) & set(doc_keywords))

        if similarity > 0.35:
            confidence = "high"
        elif similarity > 0.30:
            confidence = "moderate"
        else:
            confidence = "low"

        if rank == 1:
            prefix = "This is the most relevant document"
        elif rank <= 3:
            prefix = "This is highly relevant"
        else:
            prefix = "This is moderately relevant"

        if cluster_id == -1 or cluster_prob < threshold:
            reason = (
                f"Rank #{rank}: {prefix} based on semantic similarity "
                f"(similarity = {similarity}). This is a {confidence}-confidence match. "
                "The system used fallback mode due to low cluster confidence. "
            )
        else:
            reason = (
                f"Rank #{rank}: {prefix}. It belongs to cluster {cluster_id} "
                f"(cluster affinity = {cluster_prob}) and has semantic similarity "
                f"(similarity = {similarity}). This is a {confidence}-confidence match. "
            )

        if matched_keywords:
            reason += f"Matching keywords: {', '.join(matched_keywords[:5])}."
        else:
            reason += "Match is based on semantic meaning rather than exact keywords."

        explanations.append({
            "rank": rank,
            "cluster": cluster_id,
            "similarity": similarity,
            "cluster_affinity": cluster_prob,
            "matched_keywords": matched_keywords[:5],
            "reason": reason
        })

    return explanations