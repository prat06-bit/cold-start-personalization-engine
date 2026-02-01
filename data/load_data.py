import json
import pandas as pd
import re

def load_and_clean_data(path="newsgroups.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = list(data["content"].values())
    df = pd.DataFrame({"text": texts})

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    df["clean_text"] = df["text"].apply(clean_text)
    return df
