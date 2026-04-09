from typing import Dict, List
import reporter
import source
from models import NewsItem

from datetime import datetime
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
    incoming_news: List[NewsItem] = source.fetch_news()
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

    reporter.generate_html_report(current_cache, RUN_ID, "index.html")
    try:
        logger.print_summary()
    except:
        pass
    
if __name__ == "__main__":
    main()

