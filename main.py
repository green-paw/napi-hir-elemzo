from typing import Dict, List, Set
from checkpoint_manager import load_checkpoint, save_checkpoint
import llm_service
import reporter
import source
from models import NewsCache, NewsCluster, NewsItem

from datetime import datetime, timedelta
import builtins
from gemini_core import logger        
import numpy as np
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

    # 0.2-vel alakítunk kupacokat
    #clusters = source.create_clusters_by_embedding(list(active_cache.values()), threshold=0.2)
    
    
    densest30 = get_densest_chunk(list(active_cache.values()))
    first_anchors = llm_service.get_anchors_texts(densest30)


    return


    #mini clusters, llm névadás, trash szűrés
    #clusters = source.create_clusters_by_embedding(list(active_cache.values()), threshold=0.085)
    #processed_clusters = llm.process_mini_clusters(clusters)

    #big clusters
    #clusters = [cluster for cluster in source.create_clusters_by_embedding(list(active_cache.values()), threshold=0.25) if len(cluster.items) > 1]

    print(f"Iteratív klaszterezés indítása {len(active_cache)} hírre")
    clusters = source.iterative_clustering(list(active_cache.values()))
    clusters_len = len(clusters)

    print(f"Iteratív klaszterezés: {len(active_cache)} hír -> {len(clusters)} klaszter. LLM csoportosítás indítása")
    
    processed_clusters = llm.process_large_clusters(clusters)
    processed_clusters.sort(key=lambda c: len(c.items), reverse=True)
    clusters = [c for c in processed_clusters if len(c.items) > 1]
    print(f"{clusters_len} clusterből {clusters_len - len(clusters)} egyedi hír eldobva")
    
    #save_checkpoint("clusters.json", processed_clusters, List[NewsCluster])

    reporter.generate_html_report(processed_clusters, filename="index.html")

    try:
        logger.print_summary()
    except:
        pass

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





"""def iterative_clustering(active_cache: Dict[str, NewsItem]) -> List[NewsCluster]:
    # A hírtömeg, amiből dolgozunk
    remaining_news = list(active_cache.values())
    final_clusters: List[NewsCluster] = []

    print(f"🚀 Iteratív feldolgozás indítása {len(remaining_news)} hírrel...")

    while len(remaining_news) > 0:
        # 1. Chunk kiválasztása (az aktuális maradék elejéről)
        chunk = remaining_news[:30]
        
        # 2. Horgonyok kérése az LLM-től (Step 1)
        # Itt hívjuk a Gemini-t a chunk listájával
        anchors = get_anchors_from_llm(chunk) 
        
        if not anchors:
            # Ha ebből a 30-ból semmi nem volt fontos, eltoljuk az "ablakot"
            # de a híreket nem töröljük, hátha más horgony később behúzza őket.
            # Ha sokszor nem találunk semmit, a végén elhagyjuk őket.
            print("ℹ️ Nem találtunk új témát ebben a chunkban, ugrunk a következőre...")
            # (Itt egy 'offset' logikát alkalmazunk, hogy ne ragadjunk be)
            break # Egyelőre, a teszt kedvéért
            
        # 3. Globális szűrés a dual-anchor logikával (Matematikai rész)
        found_in_this_round = set()
        for anchor in anchors:
            # Itt történik a 0.1-es távolságú sweep_globally
            new_cluster = sweep_globally(remaining_news, anchor, threshold=0.1)
            
            if len(new_cluster.items) > 1:
                final_clusters.append(new_cluster)
                found_in_this_round.update([it.hash for it in new_cluster.items])

        # 4. Törlés a maradékból
        remaining_news = [it for it in remaining_news if it.hash not in found_in_this_round]
        
        print(f"✅ Kör kész. Talált klaszterek: {len(anchors)}, Maradék hír: {len(remaining_news)}")

    return final_clusters"""



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

def end_log():
    try:
        logger.print_summary()
    except:
        pass

if __name__ == "__main__":
    main()
    end_log()


