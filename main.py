from datetime import datetime
import output_handler
from rss_handler import fetch_news
from gemini_handler import batch_cluster_news, generate_event_summary, usage_tracker

def myPrint(message):
    """Timestampet ad minden üzenet elé (HH:MM:SS format)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    # 1. Hírek lekérése
    filtered_news = fetch_news()
    if not filtered_news:
        myPrint("Nincsenek új hírek, kilépés.")
        return
   
    # 2. Hírek formázása a klaszterező promptba (Pydantic objektumok használata)
    formatted_list = "\n".join([
        f"ID:{n.id} | CÍM: {n.title} | KIVONAT: {n.summary[:200]}" 
        for n in filtered_news
    ])

    # 3. Együttes klaszterezés (Free Flash hívás)
    myPrint(f"🧩 Szemantikus klaszterezés egyben, Free Flash modellel ({len(filtered_news)} hír)...")
    cluster_result = batch_cluster_news(formatted_list)
    
    # Események kinyerése a válaszból
    events = []
    if isinstance(cluster_result, dict):
        events = cluster_result.get("events", [])
    else:
        events = cluster_result.events or []

    myPrint(f"✅ Klaszterezés kész, {len(events)} esemény azonosítva.")

    final_data_package = []

    # 4. Végigmegyünk az azonosított eseményeken
    for i, cluster in enumerate(events, 1):
        c_ids = cluster.get("ids", []) if isinstance(cluster, dict) else getattr(cluster, "ids", [])
        
        # Kikeressük a teljes cikk objektumokat az ID-k alapján
        relevant_news_objects = [n for n in filtered_news if n.id in c_ids]
        
        if not relevant_news_objects:
            continue

        myPrint(f"  [{i}/{len(events)}] Esemény elemzése ({len(relevant_news_objects)} cikkből)...")
        
        # 5. A Lite modell hívása az összefoglalóhoz
        event_data = generate_event_summary(relevant_news_objects) 
        
        if event_data:
            # Ha Pydantic modellt kaptunk vissza a generálótól, dict-té alakítjuk
            #if hasattr(event_data, 'model_dump'):
            #    event_data = event_data.model_dump()

            # Források kinyerése az Article objektumokból
            sources_data = [
                {"name": n.source, "url": n.link} 
                for n in relevant_news_objects
            ]
            
            final_data_package.append({
                'category': event_data.category or 'EGYÉB',
                'title': event_data.title or 'Névtelen esemény',
                'summary': event_data.summary or '',
                'sources': sources_data,
                'score': event_data.score or 0
            })
        
        #time.sleep(1.2) # API kvóta védelem

    # 6. Sorbarendezés pontszám alapján csökkenő sorrendbe
    final_data_package.sort(key=lambda x: x['score'], reverse=True)

    # 7. Kimenetek (ntfy, HTML, Telegram)
    if final_data_package:
        output_handler.process_and_send(final_data_package, "")
        
    # Statisztika
    usage = usage_tracker.get_aggregated_stats()
    myPrint(f"📊 Token használat: {usage}")
    myPrint("✅ Kész.")
    
if __name__ == "__main__":
    main()