import numpy as np
from typing import List

def calculate_density(article_ids: List[int], context) -> float:
    """
    Kiszámolja a hírek sűrűségét, korrigálva az embedding modellek 
    alapértelmezett csoportosulási torzítását.
    """
    if len(article_ids) < 2: return 1.0
    
    vectors = []
    for aid in article_ids:
        if aid in context.articles and context.articles[aid].embedding:
            vectors.append(context.articles[aid].embedding)
            
    if len(vectors) < 2: return 0.0
    
    vec_array = np.array(vectors)
    
    # 1. Normalizálás (L2 norm)
    norms = np.linalg.norm(vec_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vecs = vec_array / norms
    
    # 2. Centroid hossz számítás
    centroid = np.mean(norm_vecs, axis=0)
    raw_density = float(np.linalg.norm(centroid))
    
    # 3. SKÁLÁZÁS (Offset + Gain)
    # Mivel a Gemini embeddingeknél a 0.75 körüli érték a "teljesen különböző", 
    # ezt vesszük alapnak (0.0), és az 1.0-ig tartó részt skálázzuk.
    
    offset = 0.75 # Minden, ami ezen alatt van, az 0 sűrűség
    if raw_density <= offset:
        return 0.0
    
    # A 0.75 - 1.0 tartományt kihúzzuk 0.0 - 1.0 közé
    scaled_density = (raw_density - offset) / (1.0 - offset)
    
    return min(1.0, scaled_density)