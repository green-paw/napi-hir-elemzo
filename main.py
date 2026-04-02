from typing import List
from reporter import HtmlReporter
from source import fetch_news
from datetime import datetime

import builtins
from datetime import datetime

_original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    _original_print(f"{timestamp} ", *args, **kwargs)

builtins.print = timestamped_print
from typing import List
from source import fetch_news, NewsItem
from clustering import ClusteringService
from editor import validate_and_refine_clusters
from filtering import filter_lone_wolves
from checkpoint_manager import load_checkpoint, save_checkpoint

import debugreporter

def main():
    print("🚀 Hírfeldolgozó pipeline indítása...")

    # --- 1. FÁZIS: Adatgyűjtés ---
    # Megnézzük, van-e már frissen letöltött hírünk
    news_items = load_checkpoint("raw_news.json", List[NewsItem])
    
    if not news_items:
        print("📥 Hírek letöltése az RSS feedekből...")
        news_items = fetch_news()
        save_checkpoint("raw_news.json", news_items, List[NewsItem])
    
    if not news_items:
        print("❌ Nincsenek feldolgozandó hírek. Leállás.")
        return

    # --- 2. FÁZIS: Vektorizálás és Klaszterezés ---
    # A ClusteringService magától kezeli az embeddinget és a cache-t benne
    print("📊 Matematikai klaszterezés (Mikro & Makro)...")
    service = ClusteringService(expansion_ratio=1.3, micro_threshold=0.15)
    macro_clusters, lone_wolves = service.run(news_items)

    # DEBUG GENERÁLÁS
    debug = debugreporter.DebugReporter("index.html")
    debug.generate(macro_clusters, lone_wolves)

    """
    
    # --- 3. FÁZIS: Makro-klaszterek validálása (Editor LLM) ---
    print(f"✍️ {len(macro_clusters)} makro-klaszter validálása az LLM-mel...")
    # Itt is használhatunk checkpointot, ha sok a klaszter
    validated_events = load_checkpoint("validated_events.json", List[dict])
    
    if not validated_events:
        validated_events = validate_and_refine_clusters(macro_clusters)
        save_checkpoint("validated_events.json", validated_events)

    # --- 4. FÁZIS: Magányos hírek szűrése (Filter LLM) ---
    print(f"🔍 {len(lone_wolves)} magányos hír szűrése (Zajmentesítés)...")
    important_lone_items = load_checkpoint("important_lone_items.json", List[NewsItem])
    
    if not important_lone_items:
        important_lone_items = filter_lone_wolves(lone_wolves)
        save_checkpoint("important_lone_items.json", important_lone_items, List[NewsItem])

    # --- 5. FÁZIS: Eredmények összefésülése és Megjelenítése ---
    print("\n" + "="*50)
    print("📰 VÉGLEGES HÍRÖSSZEFOGLALÓ")
    print("="*50)

    # Először a nagy események
    print(f"\n--- FŐBB ESEMÉNYEK ({len(validated_events)} db) ---")
    for ev in sorted(validated_events, key=lambda x: x['importance'], reverse=True):
        print(f"[{ev['importance']}/10] {ev['summary']}")
        print(f"   └─ Források: {', '.join(set(item.source_id for item in ev['news_items']))}\n")

    # Aztán a fontos egyedi hírek
    if important_lone_items:
        print(f"--- TOVÁBBI FONTOS HÍREK ({len(important_lone_items)} db) ---")
        for item in important_lone_items:
            print(f"• {item.title} ({item.source_id})")

    reporter = HtmlReporter("index.html")
    reporter.generate(validated_events, important_lone_items)

    print("\n" + "="*50)
    print("✅ Pipeline sikeresen lefutott.")

    """

    # Statisztika
    try:
        from gemini_core import logger        
        logger.print_summary()
    except:
        pass

if __name__ == "__main__":
    main()
