from typing import List
from reporter import DebugReporter, HtmlReporter
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
from clustering import ClusteringService, MacroCluster, get_item_profile, get_multi_anchor_vectors

def main():
    print("🚀 Hírfeldolgozó pipeline indítása...")

    # --- 1. FÁZIS: Adatgyűjtés ---
    print("📥 Hírek letöltése az RSS feedekből...")
    news_items = fetch_news()

    if not news_items:
        print("❌ Nincsenek feldolgozandó hírek. Leállás.")
        return

    # --- 2. FÁZIS: Vektorizálás és Klaszterezés ---
    # A ClusteringService magától kezeli az embeddinget és a cache-t benne
    print("📊 Matematikai klaszterezés (Mikro & Makro)...")
    service = ClusteringService(expansion_ratio=1.3, micro_threshold=0.23)
    macro_clusters, lone_wolves = service.run(news_items)

    macros = [MacroCluster(micro_clusters=m) for m in macro_clusters]

    # --- 3. FÁZIS UTÁN: SZEMANTIKUS SZŰRÉS ---
    anchors = get_multi_anchor_vectors()
    filtered_macro_clusters: List[MacroCluster] = []

    for macro in macros:
        # A reprezentáns hír (Mikró 0, Hír 0) profilja
        representative_item: NewsItem = macro.micro_clusters[0][0]
        if not representative_item.embedding: continue
        profile = get_item_profile(representative_item.embedding, anchors)
        
        # Debug info elmentése (később a HTML-be kerülhet)
        macro.profile = profile

        """
        # SZŰRÉSI LOGIKA:
        # Ha a TRASH dominál, vagy minden más túl gyenge, eldobjuk
        if profile["TRASH"] > 0.7 or max(profile["POLITICS"], profile["ECONOMY"], profile["TECH"]) < 0.4:
            print(f"🗑️ Klaszter kidobva (Zaj): {representative_item.title[:50]}...")
            continue
        """    
        filtered_macro_clusters.append(macro)

    for item in lone_wolves:
        if not item.embedding:
            continue
        item.profile = get_item_profile(item.embedding, anchors)

    # DEBUG GENERÁLÁS
    debug = DebugReporter("index.html")
    debug.generate(filtered_macro_clusters, lone_wolves)

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
