import gemini_handler
import models
import numpy as np
from typing import List, Dict
import re
from collections import Counter

class VectorMath:
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def get_centroid(embeddings: np.ndarray) -> np.ndarray:
        return VectorMath.normalize(np.mean(embeddings, axis=0))

    @staticmethod
    def get_similarities(query_vec: np.ndarray, pool_vecs: np.ndarray) -> np.ndarray:
        # Mivel normalizáltak, a dot product a koszinusz hasonlóság
        return np.dot(pool_vecs, query_vec)

def cluster_level(
    nodes: List[models.ClusterNode], 
    articles: Dict[int, models.Article],  # <--- Új paraméter
    threshold: float, 
    level: int
) -> List[models.ClusterNode]:
    
    embeddings = np.array([n.centroid for n in nodes])
    unassigned_indices = list(range(len(nodes)))
    new_level_nodes = []

    while unassigned_indices:
        # 1. Sűrűség alapú választás (Leader keresés)
        current_pool = embeddings[unassigned_indices]
        sim_matrix = np.dot(current_pool, current_pool.T)
        neighbor_counts = np.sum(sim_matrix > threshold, axis=1)
        best_local_idx = np.argmax(neighbor_counts)
        
        leader_global_idx = unassigned_indices[best_local_idx]
        leader_vec = embeddings[leader_global_idx]

        # 2. Csoport összeállítása
        all_sims = np.dot(embeddings, leader_vec)
        # Maszkolás: szabad indexek ÉS küszöb feletti hasonlóság
        member_mask = (all_sims >= threshold) & np.isin(np.arange(len(nodes)), unassigned_indices)
        group_indices = np.where(member_mask)[0]
        
        group_nodes = [nodes[i] for i in group_indices]
        
        # 3. Új Node létrehozása
        # Kigyűjtjük az összes hír-indexet, ami ehhez a csoporthoz tartozik (level 0-ig visszamenőleg)
        combined_member_indices = [idx for n in group_nodes for idx in n.member_indices]
        
        new_node = models.ClusterNode(
            level=level,
            centroid=VectorMath.get_centroid(np.array([n.centroid for n in group_nodes])),
            children=group_nodes,
            member_indices=combined_member_indices
        )
        
        # 4. KULCSSZAVAK GENERÁLÁSA (Itt hívjuk be az új logikát)
        # Most, hogy a new_node-nak már vannak gyerekei és centroidja, 
        # a get_weighted_keywords le tud fúrni a medoidig.
        new_node.summary = get_weighted_keywords(new_node, articles)
        
        new_level_nodes.append(new_node)
        
        # 5. Eltávolítjuk a kiosztott indexeket a listából
        group_indices_set = set(group_indices)
        unassigned_indices = [i for i in unassigned_indices if i not in group_indices_set]

    return new_level_nodes

def build_hierarchy(embeddings: np.ndarray, articles: dict[int, models.Article]) -> List[List[models.ClusterNode]]:
    thresholds = [0.92, 0.88, 0.85, 0.83]
    
    # 0. Szint: A levelek létrehozása
    # Itt a member_indices maga az article_id (vagy a sorrend indexe)
    current_level_nodes: List[models.ClusterNode] = []
    for i, emb in enumerate(embeddings):
        article_id = list(articles.keys())[i]
        node = models.ClusterNode(
            level=0,
            centroid=emb,
            member_indices=[article_id],
            summary=articles[article_id].title # A levél summary-je a hír címe
        )
        current_level_nodes.append(node)
    
    hierarchy: List[List[models.ClusterNode]] = [current_level_nodes]


    # L1: Szigorú (0.94)
    level_1 = cluster_level(hierarchy[0], articles, 0.94, 1)
    hierarchy.append(level_1)

    # L2: Még mindig szigorú (0.90)
    level_2 = cluster_level(hierarchy[1], articles, 0.90, 2)
    hierarchy.append(level_2)

    return hierarchy

    # --- ITT JÖN AZ ATOMBIZTOS ZSILIP ---
    # Kidobjuk a dating-et, a Bloomberg hétvégét és a verseket
    cleaned_level_2 = gemini_handler.validate_and_refine_hierarchy(level_2, articles)



    # Magasabb szintek építése
    for i, threshold in enumerate(thresholds):
        next_level = cluster_level(hierarchy[-1], articles, threshold, i + 1)
        hierarchy.append(next_level)
        
    return hierarchy

@staticmethod
def get_weighted_keywords(node: models.ClusterNode, 
                        articles: Dict[int, models.Article], 
                        top_n: int = 15) -> str:
    """
    Kiszámolja a csoport legreprezentatívabb szavait LLM nélkül, 
    a centroidhoz legközelebbi hír (medoid) súlyozásával.
    """
    # 1. Alapszavak gyűjtése és szűrése
    all_words = []
    stop_words = {
        "hogy", "vagy", "mint", "mert", "pedig", "volt", "lett", "ezer", "millió",
        "szerint", "mondta", "közölte", "alatt", "után", "miatt", "között", "belül",
        "minden", "csak", "lenne", "erről", "ebben", "annak", "lenne", "vagyis",
        "lesz", "most", "előtt", "kell", "szerint"
        "after", "with", "news", "says", "about", "from", "into", "their", "more",
        "member", "profile", "telegraph", "dating"
    }
    
    for idx in node.member_indices:
        # A címek tartalmazzák a legfontosabb entitásokat
        text = articles[idx].title.lower()
        words = re.findall(r'\b\w{4,}\b', text)
        all_words.extend([w for w in words if w not in stop_words])

    if not all_words:
        return "nincs_adat"

    # 2. Gyakoriság alapú számlálás (Helyi súly)
    counts = Counter(all_words)

    # 3. A legreprezentatívabb hír (Medoid) megkeresése
    # Lemegyünk a levelekig a legközelebbi ágon keresztül
    temp_node = node
    while not temp_node.is_leaf():
        temp_node = temp_node.get_medoid_child()
    
    # Kinyerjük a horgony szöveget az Article objektumból
    rep_article = articles[temp_node.member_indices[0]]
    representative_text = rep_article.get_short_text().lower()

    # 4. Súlyozott pontozás
    # Pont = Gyakoriság * 2.5 (ha a szó benne van a központi hírben), különben 1.0
    weighted_scores = {}
    for word, count in counts.items():
        multiplier = 2.5 if word in representative_text else 1.0
        weighted_scores[word] = count * multiplier

    # 5. Top N kiválasztása
    sorted_keywords = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ", ".join([word for word, score in sorted_keywords[:top_n]])