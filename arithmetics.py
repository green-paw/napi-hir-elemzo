from typing import List

from models import ClusterNode
import numpy as np

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

def cluster_level(nodes: List[ClusterNode], threshold: float, level: int) -> List[ClusterNode]:
    embeddings = np.array([n.centroid for n in nodes])
    unassigned_indices = list(range(len(nodes)))
    new_level_nodes = []

    while unassigned_indices:
        # 1. Sűrűség alapú választás: ki körül van a legtöbb szomszéd?
        current_pool = embeddings[unassigned_indices]
        # Hasonlósági mátrix a maradék pontok között
        sim_matrix = np.dot(current_pool, current_pool.T)
        # Megszámoljuk, melyik pontnak van a legtöbb küszöb feletti szomszédja
        neighbor_counts = np.sum(sim_matrix > threshold, axis=1)
        best_local_idx = np.argmax(neighbor_counts)
        
        # 2. Leader kijelölése
        leader_global_idx = unassigned_indices[best_local_idx]
        leader_vec = embeddings[leader_global_idx]

        # 3. Csoport összeállítása
        all_sims = np.dot(embeddings, leader_vec)
        # Csak azokat vesszük be, akik még szabadok ÉS elég közel vannak
        member_mask = (all_sims >= threshold) & np.isin(np.arange(len(nodes)), unassigned_indices)
        group_indices = np.where(member_mask)[0]
        
        group_nodes = [nodes[i] for i in group_indices]
        
        # 4. Új Node létrehozása
        new_node = ClusterNode(
            level=level,
            centroid=VectorMath.get_centroid(np.array([n.centroid for n in group_nodes])),
            children=group_nodes,
            member_indices=[idx for n in group_nodes for idx in n.member_indices]
        )
        new_level_nodes.append(new_node)
        
        # Eltávolítjuk a kiosztott indexeket
        unassigned_indices = [i for i in unassigned_indices if i not in group_indices]

    return new_level_nodes

def build_hierarchy(all_embeddings: np.ndarray, all_texts: List[str]) -> List[List[ClusterNode]]:   
    # 1. Előkészítés: Minden hír egy 0. szintű Node
    leaf_nodes = [
        ClusterNode(level=0, centroid=emb, original_text=txt, member_indices=[i]) 
        for i, (emb, txt) in enumerate(zip(all_embeddings, all_texts))
    ]

    # 2. Szintek építése (példa küszöbökkel)
    thresholds = [0.90, 0.84, 0.78, 0.72]
    current_nodes = leaf_nodes
    all_levels = [leaf_nodes]

    for i, t in enumerate(thresholds):
        if len(current_nodes) <= 5: # Megállunk, ha már elég kicsi a csúcs
            break
        current_nodes = cluster_level(current_nodes, threshold=t, level=i+1)
        all_levels.append(current_nodes)
        print(f"Level {i+1} kész: {len(current_nodes)} csoport.")

    return all_levels