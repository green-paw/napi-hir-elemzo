from typing import Dict, List, Set
from checkpoint_manager import load_checkpoint, save_checkpoint
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

def main():

    active_cache: Dict[str, NewsItem] = {}

    loaded_cache = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    trash_bin = loaded_cache.trash_bin

    sorted_batch_ids = sorted(loaded_cache.batches.keys())

    for batch_id in sorted_batch_ids:
        for h, item in loaded_cache.batches[batch_id].items():
            if h not in active_cache:
                item.downloaded = batch_id
                active_cache[h] = item

    full_blacklist = set().union(*loaded_cache.trash_bin.values())

    incoming_news: List[NewsItem] = source.fetch_news()
    for item in incoming_news:
        if not item.hash: item.compute_hash()
        if item.hash not in active_cache and item.hash not in full_blacklist:
            TextCleaner.process_single(item)
            item.downloaded = RUN_ID
            active_cache[item.hash] = item

    source.embed_news(active_cache)

    news_to_score = [
        it for it in active_cache.values() 
        if not it.profile or "POL" not in it.profile
    ]

    if news_to_score:
        anchors = source.get_anchor_embeddings()
        source.score_items(news_to_score, anchors)
        print(f"📊 {len(news_to_score)} új hír profilozva.")

    final_cache = save_flat_cache(active_cache, trash_bin)
    reporter.generate_html_report(final_cache, RUN_ID, "index.html")

    try:
        logger.print_summary()
    except:
        pass

    return





    all_live_news, current_cache = source.handle_news_feed_and_cache(incoming_news, RUN_ID)

    current_cache = source.deduplicate_to_chronological_batches(current_cache)

    if not all_live_news:
        print("❌ Nincsenek feldolgozandó hírek. Leállás.")
        return

    print(f"textCleaner előtt, {current_cache.itemCount} elem")
    TextCleaner.process(all_live_news)
    print(f"textCleaner után, {current_cache.itemCount} elem")

    anchors: Dict[str, np.ndarray] = source.get_anchor_embeddings()

    print(f"embed előtt, {current_cache.itemCount} elem")
    source.embed_news(all_live_news, current_cache, RUN_ID)
    print(f"embed után, {current_cache.itemCount} elem")
    source.score_items(all_live_news, anchors)
    source.cluster_news(all_live_news)

    trash_count = sum(1 for item in all_live_news if item.profile.get("TRASH", 0) > 0.8)
    if trash_count > 0:
        print(f"🗑️  A futás során {trash_count} hír került gyanús (trash) kategóriába.")

    source.update_current_batch(all_live_news, current_cache, RUN_ID)

    current_cache = source.deduplicate_to_chronological_batches(current_cache)

    reporter.generate_html_report(current_cache, RUN_ID, "index.html")
    try:
        logger.print_summary()
    except:
        pass
    
if __name__ == "__main__":
    main()


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
