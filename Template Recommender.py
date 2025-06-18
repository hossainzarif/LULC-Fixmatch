# STEP-BY-STEP: Template Recommendation System

# STEP 0: Install Required Packages
# pip install pandas numpy nltk sentence-transformers bertopic scikit-learn

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_distances

# STEP 2: Download NLTK Resources (only once)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# STEP 3: Load Your Dataset (Replace with your actual CSV)
df = pd.read_csv("your_dataset.csv")  # Columns: USID, ProjectName, Market, Description

# STEP 4: Normalize the Descriptions
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

df["Normalized"] = df["Description"].apply(normalize)

# STEP 5: Generate Sentence Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
df["Embedding"] = list(model.encode(df["Normalized"].tolist(), batch_size=16, show_progress_bar=True))

# STEP 6: Cluster Sentences using BERTopic
topic_model = BERTopic(embedding_model=model, nr_topics="auto")
topics, _ = topic_model.fit_transform(df["Normalized"].tolist(), embeddings=df["Embedding"].tolist())
df["TemplateID"] = topics

# STEP 7: Extract Representative Template from Each Cluster
template_reps = {}
for tid in df["TemplateID"].unique():
    cluster_df = df[df["TemplateID"] == tid]
    cluster_embeddings = np.vstack(cluster_df["Embedding"])
    center = cluster_embeddings.mean(axis=0, keepdims=True)
    dists = cosine_distances(cluster_embeddings, center)
    best_idx = dists.argmin()
    representative = cluster_df.iloc[best_idx]["Description"]
    template_reps[tid] = representative

# STEP 8: Count Frequency of Each Template in Context
grouped = (
    df.groupby(["ProjectName", "Market", "TemplateID"])
    .size()
    .reset_index(name="Count")
)

# STEP 9: Rank Templates by Frequency per (Project, Market)
grouped["Rank"] = grouped.groupby(["ProjectName", "Market"])["Count"]\
                         .rank(method="first", ascending=False)

# STEP 10: Get Top 3 Templates per Context
top_templates = grouped[grouped["Rank"] <= 3].copy()
top_templates["TemplateText"] = top_templates["TemplateID"].map(template_reps)

# STEP 11: Save or Use Output
top_templates.to_csv("top_templates_per_project_market.csv", index=False)
print(top_templates.head())




import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

# Load your dataset (replace with actual CSV)
df = pd.read_csv("your_dataset.csv")  # Ensure columns: Description, ProjectName, Market

# Normalize text
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text

df["Normalized"] = df["Description"].apply(normalize)

# Generate sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["Normalized"].tolist(), batch_size=32, show_progress_bar=True)

# Cluster using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean')
df["ClusterID"] = clusterer.fit_predict(embeddings)

# Remove noise points (ClusterID = -1)
df = df[df["ClusterID"] != -1]

# Find representative sentence for each cluster
template_reps = {}
for cid in df["ClusterID"].unique():
    cluster_df = df[df["ClusterID"] == cid]
    cluster_emb = np.vstack([embeddings[i] for i in cluster_df.index])
    center = cluster_emb.mean(axis=0, keepdims=True)
    dists = cosine_distances(cluster_emb, center)
    best_idx = dists.argmin()
    rep_sentence = cluster_df.iloc[best_idx]["Description"]
    template_reps[cid] = rep_sentence

# Group and rank templates per (ProjectName, Market)
df["TemplateText"] = df["ClusterID"].map(template_reps)

grouped = (
    df.groupby(["ProjectName", "Market", "TemplateText"])
    .size()
    .reset_index(name="Count")
)

# Rank top 3 templates per (Project, Market)
grouped["Rank"] = grouped.groupby(["ProjectName", "Market"])["Count"]\
                         .rank(method="first", ascending=False)

top_templates = grouped[grouped["Rank"] <= 3].copy()




model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["Normalized"].tolist(), batch_size=32, show_progress_bar=True)

# STEP 6: Tune HDBSCAN Cluster Size
for min_size in [10, 20, 30, 40, 50, 60]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=10, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)

    print(f"\nmin_cluster_size = {min_size}")
    print(f"→ Number of clusters: {num_clusters}")
    print(f"→ % of noise points: {noise_ratio:.2%}")

    if num_clusters > 1 and noise_ratio < 0.5:
        try:
            sil_score = silhouette_score(embeddings, labels)
            print(f"→ Silhouette score: {sil_score:.4f}")
        except:
            print("→ Silhouette score: Not computable (e.g., 1 cluster)")

# STEP 7: Choose Best Cluster Size (manual selection based on results above)
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10, metric='euclidean')
cluster_labels = clusterer.fit_predict(embeddings)
df["ClusterID"] = cluster_labels

# Remove noise points (ClusterID == -1)
df = df[df["ClusterID"] != -1]

# STEP 8: Extract Representative Sentence for Each Cluster
template_reps = {}
for cid in df["ClusterID"].unique():
    cluster_df = df[df["ClusterID"] == cid]
    cluster_emb = np.vstack([embeddings[i] for i in cluster_df.index])
    center = cluster_emb.mean(axis=0, keepdims=True)
    dists = cosine_distances(cluster_emb, center)
    best_idx = dists.argmin()
    rep_sentence = cluster_df.iloc[best_idx]["Description"]
    template_reps[cid] = rep_sentence

df["TemplateText"] = df["ClusterID"].map(template_reps)

# STEP 9: Group and Rank Templates per (ProjectName, Market)
grouped = (
    df.groupby(["ProjectName", "Market", "TemplateText"])
    .size()
    .reset_index(name="Count")
)

grouped["Rank"] = grouped.groupby(["ProjectName", "Market"])["Count"]\
                         .rank(method="first", ascending=False)

# STEP 10: Get Top 3 Templates per (Project, Market)
top_templates = grouped[grouped["Rank"] <= 3].copy()

# STEP 11: Save or Use Output
top_templates.to_csv("top_3_templates_per_project_market.csv", index=False)
print(top_templates.head())


# Save or use the result
top_templates.to_csv("top_3_templates_per_project_market.csv", index=False)
print(top_templates.head())

