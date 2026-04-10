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
    loaded_cache = NewsCache()
    #loaded_cache = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    trash_bin: Dict[str, Set[str]] = loaded_cache.trash_bin
    
    # 1. Betöltés (a tegnapi, már ellenőrzött hírek)
    for batch_id in sorted(loaded_cache.batches.keys()):
        for h, item in loaded_cache.batches[batch_id].items():
            if h not in active_cache:
                active_cache[h] = item

    full_blacklist = set().union(*trash_bin.values())

    # 2. Új hírek begyűjtése
    incoming_news: List[NewsItem] = source.fetch_news()
    newly_downloaded: List[NewsItem] = []
    
    for item in incoming_news:
        if not item.hash: item.compute_hash()
        if item.hash not in active_cache and item.hash not in full_blacklist:
            newly_downloaded.append(item)

    if newly_downloaded:
        processed_news = llm.classify_news_batch(newly_downloaded)
        for item in processed_news:
            if item.category == "TRASH":
                trash_bin.setdefault("TRASH", set()).add(item.hash)
                active_cache.pop(item.hash, None)
            else:
                TextCleaner.process_single(item)
                item.downloaded = RUN_ID
                active_cache[item.hash] = item

        print(f"✅ Feldolgozás kész. Új: {len(processed_news)} | Szűrve: {len(active_cache)}")
           
    # 3. Embedding (Már csak a tiszta hírekre!)
    #source.embed_news(active_cache)

    # TODO: 4. Klaszterezés (Centroidok alapján)
    # Ez váltja fel a "használhatatlan pontozást" logikai csoportokkal
    # source.incremental_clustering(valid_new_news, active_cache)

    # 5. Mentés és Riport
    final_cache = save_flat_cache(active_cache, trash_bin)
    reporter.generate_html_report(final_cache, RUN_ID, "index.html")
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


