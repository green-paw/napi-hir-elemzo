import numpy as np
from typing import List

def calculate_density(article_ids: List[int], context) -> float:
    if len(article_ids) < 2: return 1.0
    vectors = [context.articles[aid].embedding for aid in article_ids if context.articles[aid].embedding]
    if not vectors: return 0.0
    vec_array = np.array(vectors)
    centroid = np.mean(vec_array, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0: return 0.0
    similarities = [np.dot(v, centroid) / (np.linalg.norm(v) * centroid_norm) for v in vec_array]
    return float(np.mean(similarities))