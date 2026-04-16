from typing import Dict, List, Optional, Set
from checkpoint_manager import load_checkpoint, save_checkpoint
import gemini_core
import llm_service
import reporter
import source
from models import DualAnchor, NewsCache, NewsCluster, NewsItem

from datetime import datetime, timedelta
import builtins
from gemini_core import logger        

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from text_cleaner import TextCleaner

RUN_ID = datetime.now().isoformat()

_original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    _original_print(f"{timestamp} ", *args, **kwargs)

builtins.print = timestamped_print

llm = llm_service.LLMService()

def main():
    active_cache: Dict[str, NewsItem] = {}
    
    loaded_cache = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    #loaded_cache = NewsCache()
    trash_bin: Dict[str, Set[str]] = loaded_cache.trash_bin
    full_blacklist = set().union(*trash_bin.values()) if trash_bin else set()
    
    for batch_id in sorted(loaded_cache.batches.keys()):
        for h, item in loaded_cache.batches[batch_id].items():
            if h not in active_cache:
                active_cache[h] = item

    incoming_news: List[NewsItem] = source.fetch_news()
    newly_downloaded: List[NewsItem] = []
    
    for item in incoming_news:
        if not item.hash: item.compute_hash()
        if item.hash not in active_cache and item.hash not in full_blacklist:
            newly_downloaded.append(item)

    if newly_downloaded:
        for item in newly_downloaded:
            TextCleaner.process_single(item)
            if not item.clean_content:
                if RUN_ID not in trash_bin:
                    trash_bin[RUN_ID] = set[str]()
                trash_bin[RUN_ID].add(item.hash)
                continue
            item.downloaded = RUN_ID
            active_cache[item.hash] = item

    full_blacklist = set().union(*trash_bin.values()) if trash_bin else set()
    trash_count = len(full_blacklist)
    
    print(f"Hírek begyűjtve, active cache: {len(active_cache)} hír, trash_bin: {trash_count}")
           
    source.embed_news(active_cache)
    final_cache = save_flat_cache(active_cache, trash_bin)

    final_clusters = cluster_with_medoid_titles(list(active_cache.values()), min_cluster_size=3)
    reporter.generate_html_report(final_clusters, filename="index.html")

    return



    # 0.2-vel alakítunk kupacokat
    #clusters = source.create_clusters_by_embedding(list(active_cache.values()), threshold=0.2)

    found_hashes_in_round = set()
    final_clusters: List[NewsCluster] = []
    all_anchor_texts = []

    need_randomize = False
    chunk_size = 20

    for r in range(40):
        remaining_news = [item for item in active_cache.values() if item.hash not in found_hashes_in_round]
        items_for_anchoring = get_densest_chunk(remaining_news, chunk_size=chunk_size) if not need_randomize else np.random.choice(remaining_news, size=min(chunk_size, len(remaining_news)), replace=False).tolist()
        need_randomize = False

        anchors: List[DualAnchor] = llm_service.get_anchors_texts(items_for_anchoring)

        # embed anchors
        current_anchor_texts = []
        for a in anchors:
            all_anchor_texts.extend([a.en, a.hu])
            current_anchor_texts.extend([a.en, a.hu])

        vectors = gemini_core.embed(texts=current_anchor_texts)
        for i, anchor in enumerate(anchors):
            anchor.en_emb = vectors[i * 2]
            anchor.hu_emb = vectors[i * 2 + 1]

        # create clusters
        clusters_this_round: List[NewsCluster] = sweep_globally_winner_takes_all(remaining_news, anchors, threshold=0.0815)

        if not clusters_this_round:
            need_randomize = True
            #found_hashes_in_round.add(items_for_anchoring[0].hash)
            continue
        
        final_clusters.extend(clusters_this_round)
        for cluster in clusters_this_round:
            for item in cluster.items:
                found_hashes_in_round.add(item.hash)
        print(f"Round {r+1} finished. Clusters: {len(clusters_this_round)}, Remaining: {len(remaining_news) - sum(len(c.items) for c in clusters_this_round)}")

    final_clusters2 = finalize_clusters_semantically(final_clusters, threshold=0.03)
    print(f"Klaszter összevonás: {len(final_clusters)} -> {len(final_clusters2)}")

    reporter.generate_html_report(final_clusters2, filename="index.html")

    return

def save_flat_cache(flat_cache: Dict[str, NewsItem], trash_bin: Dict[str, Set[str]]) -> NewsCache:
    """
    Újraépíti a hierarchikus NewsCache struktúrát, kényszeríti a 24 órás limitet,
    elmenti a lemezre, és visszaadja a mentett objektumot.
    """
    new_cache = NewsCache()
    new_cache.trash_bin = trash_bin
    
    # Használjunk fix bázisidőpontot a futás alatt
    now = datetime.now()
    limit = now - timedelta(hours=24)
    
    # Statisztika a logoláshoz
    saved_count = 0
    
    for h, item in flat_cache.items():
        # Csak a limiten belüli híreket tartjuk meg
        if item.published > limit:
            # Ha valamiért nincs downloaded (pl. kézi bevitel), legyen a mostani futás
            bid = item.downloaded or now.isoformat()
            
            if bid not in new_cache.batches:
                new_cache.batches[bid] = {}
            
            new_cache.batches[bid][h] = item
            saved_count += 1
            
    # Lemezre írás
    save_checkpoint("news_feed.json", new_cache, NewsCache)
    print(f"💾 Cache mentve: {saved_count} elem {len(new_cache.batches)} batch-ben.")
    
    return new_cache

def get_densest_chunk(news_items: List[NewsItem], chunk_size: int = 30) -> List[NewsItem]:
    if len(news_items) <= chunk_size:
        return news_items
    embeddings = np.array([item.embedding for item in news_items], dtype=np.float32)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    kth_similarities = []
    for row in similarity_matrix:
        partitioned = np.partition(row, -chunk_size)
        kth_similarities.append(partitioned[-chunk_size])
    center_idx = int(np.argmax(kth_similarities))
    center_row = similarity_matrix[center_idx]
    closest_indices = np.argsort(center_row)[-chunk_size:]
    return [news_items[int(i)] for i in closest_indices]

def sweep_globally_winner_takes_all(news_items: List[NewsItem], anchors: List[DualAnchor], threshold: float = 0.1) -> List[NewsCluster]:
    # 1. Előkészítjük a klasztereket
    cluster_map: Dict[int, List[NewsItem]] = {i: [] for i in range(len(anchors))}
    
    # 2. Végigmegyünk minden egyes híren
    for item in news_items:
        if item.embedding is None:
            continue
            
        item_vec = np.array(item.embedding)
        best_dist = 1.0
        best_anchor_idx = -1
        
        # 3. Megkeressük a hírhez LEGKEVÉSBÉ TÁVOLI (legalacsonyabb hasonlóságú) horgonyt
        for idx, anchor in enumerate(anchors):
            en_vec = np.array(anchor.en_emb)
            hu_vec = np.array(anchor.hu_emb)
            
            # A hír távolsága ettől a horgonytól (min a két nyelv között)
            dist = min(1.0 - np.dot(item_vec, en_vec), 1.0 - np.dot(item_vec, hu_vec))
            
            if dist < best_dist:
                best_dist = dist
                best_anchor_idx = idx
        
        # 4. "Winner takes all": csak ha a legjobb is a threshold alatt van
        if best_anchor_idx != -1 and best_dist <= threshold:
            cluster_map[best_anchor_idx].append(item)
            
    # 5. NewsCluster objektumokká alakítjuk (csak amiben van legalább 2 hír)
    final_clusters = []
    for idx, items in cluster_map.items():
        if len(items) > 1:
            new_cluster = NewsCluster(
                cluster_id=f"M{idx}", 
                items=items
            )
            new_cluster.summary_title=anchors[idx].hu
            final_clusters.append(new_cluster)
            
            
    return final_clusters


def finalize_clusters_semantically(clusters: List[NewsCluster], threshold: float = 0.03) -> List[NewsCluster]:
    if not clusters: return []
    
    # 1. Kiszámoljuk minden klaszter középpontját (centroid)
    # Ehhez a klaszterbe tartozó hírek embeddingjeinek átlagát használjuk
    cluster_data = []
    for c in clusters:
        embeddings = np.array([it.embedding for it in c.items if it.embedding is not None])
        centroid = np.mean(embeddings, axis=0)
        # Normalizáljuk, hogy a dot product továbbra is hasonlóságot adjon
        centroid = centroid / np.linalg.norm(centroid)
        cluster_data.append({"cluster": c, "centroid": centroid, "merged": False})

    final_output = []

    for i in range(len(cluster_data)):
        if cluster_data[i]["merged"]: continue
        
        current = cluster_data[i]
        
        for j in range(i + 1, len(cluster_data)):
            if cluster_data[j]["merged"]: continue
            
            target = cluster_data[j]
            
            # Kiszámoljuk a két klaszter középpontjának távolságát
            dist = 1.0 - np.dot(current["centroid"], target["centroid"])
            
            if dist <= threshold:
                # ÖSSZEVONÁS: A target tartalmát átöntjük a currentbe
                current["cluster"].items.extend(target["cluster"].items)
                # Opcionális: a címet frissíthetjük a rövidebbre vagy az LLM-mel
                current["cluster"].summary_title += " - " + target["cluster"].summary_title
                target["merged"] = True
                
        final_output.append(current["cluster"])

    # Újraindexelés a legvégén
    for idx, c in enumerate(final_output, 1):
        c.id = f"M{idx}"
        
    return final_output









def cluster_with_medoid_titles(news_items: List[NewsItem], min_cluster_size: int = 3):
    embeddings = np.array([it.embedding for it in news_items])
    normalized_embs = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 1. Klaszterezés
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = model.fit_predict(normalized_embs)

    final_clusters = []
    for label in set(labels):
        if label == -1: continue  # Zaj kihagyása

        # Klaszterbe tartozó hírek kigyűjtése
        cluster_indices = np.where(labels == label)[0]
        cluster_items: List[NewsItem] = [news_items[int(i)] for i in cluster_indices]
        cluster_embs = normalized_embs[cluster_indices]

        # 2. Centroid számítás
        centroid = np.mean(cluster_embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # 3. MEDOID megkeresése (A hír, ami legközelebb van a centroidhoz)
        # Cosine similarity-t nézünk a centroid és a klaszter tagjai között
        similarities: np.ndarray = np.dot(cluster_embs, centroid)
        best_idx_in_cluster: int = np.argmax(similarities)
        representative_item: NewsItem = cluster_items[best_idx_in_cluster]

        # 4. Objektum létrehozása
        new_cluster = NewsCluster(
            cluster_id=f"M{label}",
            items=cluster_items,
            centroid=centroid,
            # A hír címe lesz a klaszter címe
            summary_title=representative_item.title 
        )
        final_clusters.append(new_cluster)

    return final_clusters





def end_log():
    try:
        logger.print_summary()
    except:
        pass

if __name__ == "__main__":
    main()
    end_log()


