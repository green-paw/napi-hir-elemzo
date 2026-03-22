import json
from google.genai import Client
import config
import shared_state
from rss_engine import fetch_all_news # a fájlt és a függvényt át kell nevezni!
from llm_core import setup_gemini_cache, cleanup_cache
from classifier import discover_rolling_topics, refine_to_top_30, classify_news_with_lite, clean_clusters
from summarizer import generate_final_reports
from html_exporter import export_to_html
import os

def run_news_pipeline():
    # 1. Kliensek inicializálása
    client_main = Client(api_key=config.GOOGLE_API_KEY_MAIN)
    client_free = Client(api_key=config.GOOGLE_API_KEY_FREE)

    try:
        print("📥 1. Hírek letöltése...")
        shared_state.filtered_news = fetch_all_news(config.RSS_FEEDS)
        
        # JSON szöveg generálása a cache-hez
        news_json = json.dumps([n.model_dump() for n in shared_state.filtered_news], default=str)

        print("🧠 2. Cache inicializálása...")
        setup_gemini_cache(client=client_main, formatted_json_text=news_json)

        print("🔍 3. Témák felderítése (Flash)...")
        raw_topics = discover_rolling_topics(client=client_main)
        
        print("🎯 4. Top 30 téma kiválasztása (Flash)...")
        shared_state.master_topics = refine_to_top_30(client=client_main, raw_topics=raw_topics)

        print("🗂️ 5. Hírek besorolása (Flash Lite)...")
        raw_clusters = classify_news_with_lite(client=client_main)

        print("🧹 6. Klaszterek tisztítása és darabolása...")
        valid_clusters = clean_clusters(raw_clusters=raw_clusters, min_news=3)

        print("✍️ 7. Összefoglalók írása (Free Flash)...")
        final_reports = generate_final_reports(client=client_free, valid_clusters=valid_clusters)

        output_filename: str = os.getenv("OUTPUT_FILENAME", "index.html")

        print(f"🌐 8. HTML generálása és mentése ide: {output_filename}...")
        export_to_html(summaries=final_reports, filename=output_filename)

        print("✅ Kész! Sikeres feldolgozás.")
        return True

    except Exception as e:
        print(f"❌ Végzetes hiba a folyamatban: {e}")
    finally:
        cleanup_cache(client=client_main)

if __name__ == "__main__":
    run_news_pipeline()