import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple
import gemini_core
from source import NewsItem

from checkpoint_manager import load_checkpoint, save_checkpoint

class ClusteringService:
    def __init__(self, expansion_ratio: float = 2.0, micro_threshold: float = 0.1):
        """
        expansion_ratio: Hányszorosára nőhet a sűrűség (elemszám) a makro körben.
        micro_threshold: A szigorú távolság az azonos hírek (duplikációk) kiszűréséhez.
        """
        self.expansion_ratio = expansion_ratio
        self.micro_threshold = micro_threshold

    @staticmethod
    def _get_top_half_avg_size(labels: np.ndarray) -> float:
        """Kiszámolja a legalább 2 elemű klaszterek felső felének átlagos méretét."""
        unique, counts = np.unique(labels, return_counts=True)
        # Csak a valódi csoportokat nézzük, az egyedülálló híreket (zaj) nem
        cluster_sizes = counts #[counts > 1]
        
        if len(cluster_sizes) == 0:
            return 1.0
            
        cluster_sizes.sort()
        # Felső fele egészosztással (//2)
        top_half = cluster_sizes[len(cluster_sizes) // 2:]
        return float(np.mean(top_half))

    def _prepare_embeddings(self, news_items: List[NewsItem]) -> np.ndarray:
        """Biztosítja, hogy minden hírnek legyen vektora, majd visszaadja őket np.array-ként."""
        items_to_embed = [item for item in news_items if item.embedding is None]
        
        if items_to_embed:
            print(f"🧠 {len(items_to_embed)} hír vektorizálása folyamatban...")
            texts = [f"{item.title}. {item.content}" for item in items_to_embed]
            vectors = gemini_core.embed(texts, task_type="CLUSTERING")
            
            for item, vector in zip(items_to_embed, vectors):
                item.embedding = vector
                
        return np.array([item.embedding for item in news_items])

def run(self, news_items: List[NewsItem]) -> Tuple[List[List[List[NewsItem]]], List[NewsItem]]:
        if not news_items:
            return [], []

        cached_items = load_checkpoint("news_items_with_embeddings.json", List[NewsItem])
        if cached_items:
            # Szinkronizáljuk a vektorokat az aktuális listával (ha pl. új hírek jöttek azóta)
            # De egyszerűbb, ha a teljes listát cseréljük a cache-eltre:
            news_items = cached_items
            print("✅ Vektorok betöltve a NewsItem cache-ből.")
        else:
            # Ha nincs cache, generálunk és mentünk
            self._prepare_embeddings(news_items)
            save_checkpoint("news_items_with_embeddings.json", news_items, List[NewsItem])

        # A klaszterezéshez szükségünk van a mátrixra a NewsItem-ekből
        embeddings = np.array([item.embedding for item in news_items])
        
        # --- 1. MIKRO KÖR ---
        micro = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.micro_threshold,
            linkage='average'
        ).fit(embeddings)
        
        # Mikro-szintű csoportosítás (label -> List[NewsItem])
        micro_groups: Dict[int, List[NewsItem]] = {}
        for idx, m_label in enumerate(micro.labels_):
            micro_groups.setdefault(m_label, []).append(news_items[idx])

        # Sűrűség számítás a mikro-klaszterek alapján
        micro_density = self._get_top_half_avg_size(micro.labels_)
        target_density = micro_density * self.expansion_ratio
        
        # --- 2. MAKRO KÖR ---
        # A loop ugyanúgy fut a teljes embedding mátrixon a cél-sűrűségig
        current_threshold = self.micro_threshold + 0.05
        best_macro_labels = micro.labels_
        
        while current_threshold < 0.7:
            macro = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=current_threshold,
                linkage='average'
            ).fit(embeddings)
            
            if self._get_top_half_avg_size(macro.labels_) >= target_density:
                best_macro_labels = macro.labels_
                break
            best_macro_labels = macro.labels_
            current_threshold += 0.05

        # --- 3. HIERARCHIKUS ÖSSZEÁLLÍTÁS ---
        # macro_id -> { micro_id -> [NewsItems] }
        hierarchy: Dict[int, Dict[int, List[NewsItem]]] = {}
        
        for idx, (m_label, macro_label) in enumerate(zip(micro.labels_, best_macro_labels)):
            if macro_label not in hierarchy:
                hierarchy[macro_label] = {}
            if m_label not in hierarchy[macro_label]:
                hierarchy[macro_label][m_label] = []
            hierarchy[macro_label][m_label].append(news_items[idx])

        final_macro_clusters = []
        lone_wolves = []

        for macro_id, micros in hierarchy.items():
            # Ha a makro-klaszterben csak egy mikro-klaszter van, és az is csak 1 hír
            all_items_in_macro = [item for micro_list in micros.values() for item in micro_list]
            
            if len(all_items_in_macro) > 1:
                # Ez egy valódi makro-klaszter, amiben mikro-klaszterek listája van
                final_macro_clusters.append(list(micros.values()))
            else:
                lone_wolves.append(all_items_in_macro[0])

        return final_macro_clusters, lone_wolves