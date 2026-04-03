from typing import List
import editor
import gemini_core
import reporter
import source
from source import NewsItem
from clustering import ClusteringService, MacroCluster, get_item_profile, get_multi_anchor_vectors

from datetime import datetime
import builtins

_original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    _original_print(f"{timestamp} ", *args, **kwargs)

builtins.print = timestamped_print
from typing import List

def main():
    print("🚀 Hírfeldolgozó pipeline indítása...")

    # --- 1. FÁZIS: Adatgyűjtés ---
    print("📥 Hírek letöltése az RSS feedekből...")
    news_items = source.fetch_news()

    if not news_items:
        print("❌ Nincsenek feldolgozandó hírek. Leállás.")
        return

    # --- 2. FÁZIS: Vektorizálás és Klaszterezés ---
    # A ClusteringService magától kezeli az embeddinget és a cache-t benne
    print("📊 Matematikai klaszterezés (Mikro & Makro)...")

    anchors = get_multi_anchor_vectors()

    service = ClusteringService(expansion_ratio=1.3, micro_threshold=0.35)
    service._prepare_embeddings(news_items)
    for i in news_items:
        if not i.embedding: continue
        i.profile = get_item_profile(i.embedding, anchors)

    before = len(news_items)
    news_items = [i for i in news_items if i.profile["NET_RELEVANCE"] > 1]
    print(f"{before} hírből filterelés után maradt {len(news_items)}")

    macro_clusters, lone_wolves = service.run(news_items)
    macros = [MacroCluster(micro_clusters=m) for m in macro_clusters]

    editor.process_macros_parallel(macros)

    #for i, m in enumerate(macros, 1):
    #    print(f"Makró név generálás {i}/{len(macros)}")
    #    editor.generate_macro_label(m)

    items_to_embed = [item for item in macros if item.embedding is None]
    
    if items_to_embed:
        print(f"🧠 {len(items_to_embed)} makró vektorizálása folyamatban...")
        texts = [item.title for item in items_to_embed]
        vectors = gemini_core.embed(texts, task_type="CLUSTERING")
        
        if len(vectors) == len(items_to_embed):
            for item, vector in zip(items_to_embed, vectors):
                item.embedding = vector
        else:
            print("⚠️ Hiba: A kapott vektorok száma nem egyezik a kéréssel!")
            for macro in items_to_embed:
                representative_micro = max(macro.micro_clusters, key=len)
                if representative_micro and len(representative_micro) > 0:
                    macro.embedding = representative_micro[0].embedding            

    for macro in macros:
        if not macro.embedding: continue
        macro.profile = get_item_profile(macro.embedding, anchors)

    # DEBUG GENERÁLÁS
    debug = reporter.DebugReporter("index.html")
    debug.generate(macros, lone_wolves)

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
