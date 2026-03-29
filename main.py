import config
import rss_handler
import ingestion
import orchestrator
import gemini_core
import output_handler
from models import TokenLogger

import builtins
from datetime import datetime

_original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    _original_print(f"{timestamp} ", *args, **kwargs)

builtins.print = timestamped_print

import config
from google.genai import Client

def clear_all_caches():
    client = Client(api_key=config.GOOGLE_API_KEY)
    print("🧹 Aktív cache-ek keresése...")
    
    try:
        # Kilistázzuk az összes meglévő cache-t
        active_caches = client.caches.list()
        
        count = 0
        for c in active_caches:
            print(f"🗑️ Törlés: {c.display_name} ({c.name})...")
            client.caches.delete(name=c.name)
            count += 1
            
        if count == 0:
            print("✨ Nem találtam törlendő cache-t.")
        else:
            print(f"✅ Összesen {count} cache törölve.")
            
    except Exception as e:
        print(f"❌ Hiba a takarítás során: {e}")

if __name__ == "__main__":
    clear_all_caches()






def main():
    print("--- 🚀 AI Hírszerzési Rendszer Indítása ---")
    
    # 1. HÍREK BEGYŰJTÉSE (RSS + Checkpoint kezelés)
    # Az rss_handler elvégzi a duplikációs szűrést és az auto-increment ID kiosztást
    news_pool = rss_handler.fetch_news()
    
    if not news_pool:
        print("❌ Nincs feldolgozható hír. Kilépés.")
        return

    # 2. SESSION FELÉPÍTÉSE (Ingestion)
    # Ez a lépés végzi el az embeddingeket és hozza létre a Context Cache-t (32k+ token esetén)
    # A gemini_core.setup_gemini_cache-t hívja meg belülről
    context = ingestion.create_session(news_pool, config.GOOGLE_API_KEY)
    
    try:
        # 3. REKURZÍV ELEMZÉS (Orchestrator)
        # Elindítjuk a folyamatot a teljes hírlistával (gyökér szint)
        all_ids = list(context.articles.keys())
        print(f"🧠 Elemzés indítása {len(all_ids)} hírrel a '{config.MODEL_LITE_ID}' modellel...")
        
        # A rekurzió felépíti a ReportNode hierarchiát
        context.report_root = orchestrator.recursive_orchestrator(
            current_ids=all_ids, 
            path_nodes=[], 
            context=context
        )
        
        # 4. KIMENET GENERÁLÁSA (Output Handler)
        # Az új, hierarchikus HTML jelentés elkészítése a források visszafejtésével
        output_handler.create_report(context)
        
    except Exception as e:
        print(f"💥 Kritikus hiba az elemzés során: {e}")
        
    finally:
        # 5. TAKARÍTÁS (Cleanup)
        # Bármi történik, a futás végén töröljük a cache-t a Google szervereiről
        if context.cache_id:
            gemini_core.cleanup_cache(context.client, context.cache_id)
        
        # 6. STATISZTIKA
        # A TokenLogger segítségével kiírjuk a becsült költségeket és token használatot
        print("\n--- 📊 Futási Statisztika ---")
        _print_execution_summary(context.logger)
        print("--- ✅ Folyamat befejezve ---")

def _print_execution_summary(logger: TokenLogger):
    """Segédfüggvény a logolt tokenek összesítésére."""
    total_input = sum(log["input"] for log in logger.log)
    total_output = sum(log["output"] for log in logger.log)
    total_cached = sum(log["cached"] for log in logger.log)
    
    print(f"📥 Összes bemeneti token: {total_input}")
    print(f"📤 Összes kimeneti token: {total_output}")
    print(f"💾 Ebből cache-elt: {total_cached}")
    print(f"🔄 API hívások száma: {len(logger.log)}")

#if __name__ == "__main__":
#    main()