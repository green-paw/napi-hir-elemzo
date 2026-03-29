import numpy as np
from typing import List

def calculate_density(article_ids: List[int], context) -> float:
    """
    Kiszámolja, mennyire szoros a kapcsolat a hírek között.
    0.0 = teljesen különböző témák, 1.0 = szinte azonos szövegek.
    """
    if len(article_ids) < 2: return 1.0
    
    # Csak azokat a vektorokat gyűjtjük be, amik léteznek
    vectors = []
    for aid in article_ids:
        if aid in context.articles and context.articles[aid].embedding:
            vectors.append(context.articles[aid].embedding)
            
    if len(vectors) < 2: return 0.0
    
    vec_array = np.array(vectors)
    
    # 1. Normalizáljuk a vektorokat (hogy a hosszuk 1 legyen)
    norms = np.linalg.norm(vec_array, axis=1, keepdims=True)
    # Kerüljük a nullával való osztást
    norms[norms == 0] = 1.0
    norm_vecs = vec_array / norms
    
    # 2. Kiszámoljuk a Koszinusz-hasonlósági mátrixot (mindenki mindenkivel)
    # Ha túl sok hír van (pl. 700+), a teljes mátrix lassú lenne, 
    # ezért mintát veszünk vagy a centroid-távolság szórását nézzük.
    
    centroid = np.mean(norm_vecs, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    
    if centroid_norm < 0.001: return 0.0
    
    # Mennyire szóródnak a vektorok az átlagtól?
    # Egy sűrű csoportnál a centroid hossza közel lesz az 1-hez.
    # Egy szétzilált csoportnál a centroid hossza közel lesz a 0-hoz.
    return float(centroid_norm)