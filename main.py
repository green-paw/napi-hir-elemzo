import os
from typing import Dict, List, Set, TypeVar
from analyzer import AnalysisResult, MacroAnalysisPair, analyze_macro_cluster
from checkpoint_manager import CacheWrapper, NewsCache, load_checkpoint, save_checkpoint
import clustering
import editor
import reporter
import source
from source import NewsItem
import source

from clustering import (
    ClusteringService, 
    get_multi_anchor_vectors, 
    get_item_profile, 
    ensure_item_embeddings, 
    ensure_macro_embeddings,
    get_taxonomy_suggestion
)

from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timedelta
import builtins
from gemini_core import logger        
from pydantic import BaseModel
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

    if not all_live_news:
        print("❌ Nincsenek feldolgozandó hírek. Leállás.")
        return

    TextCleaner.process(all_live_news)

    first_item = all_live_news[0]
    clean_txt = first_item.clean_content if first_item.clean_content else "Nincs tiszta szöveg"

    print(f"DEBUG: {first_item.id} tiszta szövege: {clean_txt[:50]}...")

    anchors: Dict[str, np.ndarray] = clustering.get_anchor_embeddings()

    source.embed_news(all_live_news, current_cache, RUN_ID)
    source.score_items(all_live_news, anchors)
    source.cluster_news(all_live_news)

    trash_count = sum(1 for item in all_live_news if item.profile.get("TRASH", 0) > 0.8)
    if trash_count > 0:
        print(f"🗑️  A futás során {trash_count} hír került gyanús (trash) kategóriába.")

    source.update_current_batch(all_live_news, current_cache, RUN_ID)


    # 2. Hírek vektorizálása és profilozása
    ensure_item_embeddings(news_items)
    for i in news_items:
        if i.embedding:
            i.profile = get_item_profile(i.embedding, anchors)

    # 3. Szemét szűrése a nyers hírekből
    before = len(news_items)
    news_items = [i for i in news_items if i.profile.get("NET_RELEVANCE", 0) > 1.0]
    print(f"Szűrés: {before} hírből maradt {len(news_items)}")

    # 4. Makró klaszterek építése
    service = ClusteringService(expansion_ratio=1.5, micro_threshold=0.35)
    macros, lone_wolves = service.build_macros(news_items)

    # 5. LLM Cím és Impact Score generálása
    editor.generate_macro_labels_parallel(macros)

    # 6. Makrók vektorizálása (a letisztított címek alapján) és profilozása
    ensure_macro_embeddings(macros)
    for m in macros:
        if m.embedding:
            m.profile = get_item_profile(m.embedding, anchors)

    # 7. Szigorú szűrés a "Top" makrókra (Súlyozott képlet)
    top_macros = [
        m for m in macros 
        if m.impact is not None and m.score >= 9.0
    ]
    print(f"Top makrók kiválasztva: {len(top_macros)} / {len(macros)}")

    secondary_macros = [m for m in macros if m not in top_macros]

    #tax = get_taxonomy_suggestion(top_macros)
    #print(tax)

    for i, m in enumerate(top_macros): m.id = i

    print(f"Analízis indul {len(top_macros)} makróra")
    results: List[MacroAnalysisPair] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(analyze_macro_cluster, top_macros))

    fileName = os.getenv("CURRENT_BRANCH", "main")
    if fileName == "main":
        fileName = "index"
    reporter.generate_analysis_html(results, f"{fileName}.html")

    # 8. Csak a top makrókból építünk mega klasztereket (Témaköröket)
    mega_clusters = service.build_megas_with_llm(top_macros)

    # 9. Debug HTML generálása
    debug = reporter.DebugReporter("debug.html")
    debug.generate(mega_clusters, secondary_macros, lone_wolves)


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
        logger.print_summary()
    except:
        pass

if __name__ == "__main__":
    main()

