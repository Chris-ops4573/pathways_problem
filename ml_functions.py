import hdbscan
import numpy as np
import textstat

def extract_hdbscan_features(embeddings: list[list[float]]) -> dict:
    # Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    clusterer.fit(embeddings)

    labels = clusterer.labels_  # Cluster labels: -1 means outlier
    outlier_scores = clusterer.outlier_scores_

    total_points = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)

    largest_cluster_size = 0
    if n_clusters > 0:
        largest_cluster_size = np.max(np.bincount(labels[labels != -1]))

    largest_cluster_ratio = largest_cluster_size / total_points
    avg_outlier_score = outlier_scores[labels == -1].mean() if n_outliers > 0 else 0.0

    return [
        n_clusters,
        float(largest_cluster_ratio),
        float(avg_outlier_score)
    ]

def compute_readability_score(text: str) -> float:
    try:
        return textstat.flesch_reading_ease(text)
    except Exception as e:
        print(f"Error computing readability: {str(e)}")
        return 0.0  # fallback value if something goes wrong
