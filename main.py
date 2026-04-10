from typing import Dict, List, Set
from checkpoint_manager import load_checkpoint, save_checkpoint
import llm_service
import reporter
import source
from models import NewsCache, NewsItem

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
    
    # ha törölni akarom a cache-t:
    #loaded_cache = NewsCache()
    loaded_cache = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    trash_bin: Dict[str, Set[str]] = loaded_cache.trash_bin
    full_blacklist = set().union(*trash_bin.values())

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
            item.downloaded = RUN_ID
            active_cache[item.hash] = item
           
    source.embed_news(active_cache)
    final_cache = save_flat_cache(active_cache, trash_bin)

    clusters = source.create_clusters_by_embedding(list(active_cache.values()), threshold=0.3)
    reporter.generate_html_report(clusters, filename="index.html")

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








if __name__ == "__main__":
    main()


