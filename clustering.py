import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import gemini_core
from source import NewsItem
from checkpoint_manager import load_checkpoint, save_checkpoint

# --- ADATSTRUKTÚRÁK ---

@dataclass
class MacroCluster:
    micro_clusters: List[List[NewsItem]]
    profile: Dict[str, float] = field(default_factory=dict)
    title: str = ""
    embedding: Optional[List[float]] = None
    impact: int = 0

    @property
    def score(self, weight: float = 0.3) -> float:
        try:
            return float(self.impact) * (1 + weight) + self.profile.get('NET_RELEVANCE', 0.0) * (1 - weight)
        except Exception:
            return 0.0

@dataclass
class MegaCluster:
    macros: List[MacroCluster]
    title: str = ""
    
    @property
    def score(self) -> float:
        """A Mega-klaszter fontossága a benne lévő makrók átlagos pontszáma alapján."""
        if not self.macros: return 0.0
        return sum(m.score for m in self.macros) / len(self.macros)


# --- VEKTORIZÁCIÓS SEGÉDFÜGGVÉNYEK ---

def ensure_item_embeddings(news_items: List[NewsItem]) -> None:
    """Biztosítja, hogy minden NewsItem-nek legyen embeddingje (Cache támogatással)."""
    items_to_embed = [item for item in news_items if item.embedding is None]
    
    if items_to_embed:
        print(f"🧠 {len(items_to_embed)} hír vektorizálása folyamatban...")
        texts = [f"{item.title}. {item.content}" for item in items_to_embed]
        vectors = gemini_core.embed(texts, task_type="CLUSTERING")
        
        for item, vector in zip(items_to_embed, vectors):
            item.embedding = vector

def ensure_macro_embeddings(macros: List[MacroCluster]) -> None:
    """Biztosítja a Makrók embeddingjét a címük alapján, fallback logikával."""
    items_to_embed = [item for item in macros if item.embedding is None]
    
    if items_to_embed:
        print(f"🧠 {len(items_to_embed)} makró vektorizálása folyamatban...")
        texts = [item.title for item in items_to_embed]
        vectors = gemini_core.embed(texts, task_type="CLUSTERING")
        
        if len(vectors) == len(items_to_embed):
            for item, vector in zip(items_to_embed, vectors):
                item.embedding = vector
        else:
            print("⚠️ Hiba: A kapott vektorok száma nem egyezik a kéréssel! Fallback aktiválva.")
            for macro in items_to_embed:
                representative_micro = max(macro.micro_clusters, key=len, default=[])
                if representative_micro and representative_micro[0].embedding:
                    macro.embedding = representative_micro[0].embedding


# --- KLASZTEREZŐ SZOLGÁLTATÁS ---

class ClusteringService:
    def __init__(self, expansion_ratio: float = 1.3, micro_threshold: float = 0.35, mega_threshold: float = 0.65):
        self.expansion_ratio = expansion_ratio
        self.micro_threshold = micro_threshold
        self.mega_threshold = mega_threshold

    @staticmethod
    def _get_top_half_avg_size(labels: np.ndarray) -> float:
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = counts
        if len(cluster_sizes) == 0: return 1.0
        cluster_sizes.sort()
        top_half = cluster_sizes[len(cluster_sizes) // 2:]
        return float(np.mean(top_half))

    def build_macros(self, news_items: List[NewsItem]) -> Tuple[List[MacroCluster], List[NewsItem]]:
        """A már vektorizált hírekből épít MacroCluster-eket és magányos farkasokat."""
        if not news_items: return [], []

        embeddings = np.array([item.embedding for item in news_items])
        
        # 1. MIKRO KÖR
        micro = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.micro_threshold, linkage='average'
        ).fit(embeddings)
        
        micro_density = self._get_top_half_avg_size(micro.labels_)
        target_density = micro_density * self.expansion_ratio
        
        # 2. MAKRO KÖR
        current_threshold = self.micro_threshold + 0.05
        best_macro_labels = micro.labels_
        
        while current_threshold < 0.7:
            macro = AgglomerativeClustering(
                n_clusters=None, distance_threshold=current_threshold, linkage='average'
            ).fit(embeddings)
            
            if self._get_top_half_avg_size(macro.labels_) >= target_density:
                best_macro_labels = macro.labels_
                break
            best_macro_labels = macro.labels_
            current_threshold += 0.05

        # 3. HIERARCHIKUS ÖSSZEÁLLÍTÁS
        hierarchy: Dict[int, Dict[int, List[NewsItem]]] = {}
        for idx, (m_label, macro_label) in enumerate(zip(micro.labels_, best_macro_labels)):
            hierarchy.setdefault(macro_label, {}).setdefault(m_label, []).append(news_items[idx])

        macros = []
        lone_wolves = []

        for macro_id, micros in hierarchy.items():
            all_items_in_macro = [item for micro_list in micros.values() for item in micro_list]
            if len(all_items_in_macro) > 1:
                # Közvetlenül MacroCluster objektumként adjuk vissza
                macros.append(MacroCluster(micro_clusters=list(micros.values())))
            else:
                lone_wolves.append(all_items_in_macro[0])

        return macros, lone_wolves

    def build_megas(self, macros: List[MacroCluster]) -> List[MegaCluster]:
        """A leszűrt, vektorizált makrókból nagy témaköröket (Mega) épít."""
        valid_macros = [m for m in macros if m.embedding is not None]
        if not valid_macros: return []

        embeddings = np.array([m.embedding for m in valid_macros])
        
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='cosine', # Nagyon fontos a megfelelő távolságmérés! (Régebbi sklearn esetén: affinity='cosine')
            linkage='complete',
            distance_threshold=0.2  # 0.65 helyett 0.30 - ez sokkal finomabb vágást csinál
        ).fit(embeddings)

        mega_dict: Dict[int, List[MacroCluster]] = {}
        for idx, label in enumerate(clustering.labels_):
            mega_dict.setdefault(label, []).append(valid_macros[idx])

        megas = []
        for m_list in mega_dict.values():
            sorted_macros = sorted(m_list, key=lambda x: x.score, reverse=True)
            megas.append(MegaCluster(macros=sorted_macros))

        return sorted(megas, key=lambda x: x.score, reverse=True)


# --- PROFILOZÓ FUNKCIÓK ---

ANCHOR_DEFINITIONS = {
    "POLITICS": "Government, elections, legislation, diplomacy, international relations, political parties, Tisza, Fidesz, Orbán, Magyar Péter, kém, választás, titkosszolgálat",
    "ECONOMY": "Markets, finance, inflation, central banks, trade, corporate earnings, GDP, taxes, gazdaság, infláció, olajár, üzemanyag, energia.",
    "TECH": "Space exploration, NASA, AI, software, hardware, scientific breakthroughs, engineering.",
    "TRASH": "Dating profiles, celebrity gossip, recipes, horoscopes, social media fluff, lottery, daily weather, member profiles for dating, dating advertisements, personal introduction, non-news content, user accounts, age and gender tags, lifestyle fluff."
}

def get_multi_anchor_vectors() -> Dict[str, np.ndarray]:
    anchor_cache_file = "multi_anchors.json"
    cached = load_checkpoint(anchor_cache_file, Dict[str, List[float]])
    if cached:
        return {k: np.array(v).reshape(1, -1) for k, v in cached.items()}

    print("⚓ Többirányú horgony-vektorok generálása...")
    keys = list(ANCHOR_DEFINITIONS.keys())
    texts = list(ANCHOR_DEFINITIONS.values())
    vectors = gemini_core.embed(texts, task_type="RETRIEVAL_QUERY")
    
    anchor_dict = {keys[i]: vectors[i] for i in range(len(keys))}
    save_checkpoint(anchor_cache_file, anchor_dict, Dict[str, List[float]])
    return {k: np.array(v).reshape(1, -1) for k, v in anchor_dict.items()}

def scale_score(raw_score: float, v_min: float = 0.4, v_max: float = 0.8) -> float:
    scaled = (raw_score - v_min) / (v_max - v_min)
    return round(float(np.clip(scaled, 0.0, 1.0)), 2)

def get_item_profile(item_embedding: List[float], anchors: Dict[str, np.ndarray]) -> Dict[str, float]:
    item_v = np.array(item_embedding).reshape(1, -1)
    profile = {}
    for name, anchor_v in anchors.items():
        score = cosine_similarity(item_v, anchor_v)[0][0]
        profile[name] = scale_score(score) * 10
    profile_max = max(profile["POLITICS"], profile["ECONOMY"], profile["TECH"]) 
    profile["NET_RELEVANCE"] = profile_max - profile["TRASH"]
    return profile
