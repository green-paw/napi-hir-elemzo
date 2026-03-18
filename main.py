# saját importok
import config
import output_handler

# Csak a szükséges handler funkciók
from gemini_handler import (
    get_strategic_topics, validate_news_clusters, 
    generate_event_summary, get_gemini_embeddings, 
    translate_if_needed, MultiClusterResponse,
    refine_event_list
)

from rss_handler import fetch_news

# általános importok
import math
import time
from concurrent.futures import ThreadPoolExecutor # A gyors fordításhoz
from sklearn.cluster import AgglomerativeClustering

from datetime import datetime
import random

def myPrint(message):
    """Timestampet ad minden üzenet elé (HH:MM:SS format)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# --- Szemantikus szűrő matematikai alapjai ---
def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0

def calculate_priority_score(event):
    """Kiszámolja a végleges 0-100-as prioritási pontszámot a hibrid modell alapján."""
    # Mivel az event egy dict, így ellenőrizzük:
    if not event or 'scores' not in event:
        return 0

    scores = event.get('scores', {})
    llm_base = (scores.get('impact', 0) * 0.6) + (scores.get('relevance', 0) * 0.3) + (scores.get('novelty', 0) * 0.1)
    cluster_size = len(event.get('ids', []))
    size_bonus = min(cluster_size * 0.8, 25.0) 
    final_score = (llm_base * 7.5) + size_bonus
    
    return min(round(final_score), 100)
    
def semantic_filter(news_pool, topics, top_p=0.85):
    if not topics or not news_pool: return news_pool
    myPrint(f"🔍 Szemantikus rangsorolás: {len(news_pool)} hír...")
    
    topic_embs = get_gemini_embeddings(topics)
    
    # 1. SZUPER-SZÖVEG generálása az embeddinghez (mindent beleteszünk, ami számít)
    news_texts = [
        f"CÍM: {n['title']} CÍMKÉK: {', '.join(n.get('tags', []))} KIVONAT: {n.get('summary', '')[:200]}" 
        for n in news_pool
    ]
    news_embs = get_gemini_embeddings(news_texts)
    
    # 2. Relevancia számítása és VEKTOROK MENTÉSE
    for i, n_emb in enumerate(news_embs):
        # Eltároljuk a vektort, hogy a klaszterezőnél már ne kelljen API-t hívni!
        news_pool[i]['embedding'] = n_emb
        
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        news_pool[i]['match_score'] = max(sims) if sims else 0

    filtered = [n for n in news_pool if n.get('match_score', 0) > top_p]
    filtered.sort(key=lambda x: x['match_score'], reverse=True)
    
    #final_selection = filtered[:top_k]
    final_selection = filtered
    
    myPrint(f"✅ Rangsorolás kész: {len(final_selection)} hír továbbküldve (átlagos relevancia: {sum(n['match_score'] for n in final_selection)/len(final_selection) if final_selection else 0:.2f})")
    
    return final_selection

def cluster_news(news_pool):
    if not news_pool: return []
    myPrint(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # 1. VEKTOROK ÚJRAHASZNOSÍTÁSA (Nincs API hívás, instant lefut!)
    embeddings = [n['embedding'] for n in news_pool]
    groups = auto_cluster(embeddings, news_pool)
    total_raw_groups = len(groups)

    clusters_to_validate = []
    discarded_summaries = []
    for label, news_list in groups.items():
        count = len(news_list)
        avg_relevance = sum(n.get('relevance_score', 0) for n in news_list) / count
        
        if count >= 3:
            clusters_to_validate.append((label, news_list))
        elif count == 2 and avg_relevance > 0.92:
            clusters_to_validate.append((label, news_list))
        elif count == 1 and avg_relevance > 0.97:
            clusters_to_validate.append((label, news_list))
        else:
            discarded_summaries.append(summary_text)
            continue
    
    num_to_process = len(clusters_to_validate)
    myPrint(f"📉 Szűrés után {num_to_process}/{total_raw_groups} klaszter maradt validálásra.")
    
    final_clusters = []
    
    # 2. VALIDÁCIÓS CIKLUS
    for i, (label, items) in enumerate(clusters_to_validate, 1):
        # TOKEN-SPÓROLÁS: Csak az első 15 legrelevánsabb hírt mutatjuk meg a Lite-nak
        representative_items = items[:15]
        
        formatted_list = "\n".join([
            f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}" 
            for n in representative_items
        ])

        myPrint(f"  [{i}/{num_to_process}] Lite validáció | Klaszter ID: {label} | {len(items)} hír (ebből {len(representative_items)} küldve)...")
        
        # A validáló hívás
        result = validate_news_clusters(formatted_list)

        # 3. EREDMÉNYEK FELDOLGOZÁSA ÉS ID-K VISSZAÍRÁSA
        events = []
        if isinstance(result, dict):
            events = result.get("events", [])
        elif hasattr(result, "events"):
            events = result.events

        # A Lite csak 15 ID-t látott, de mi az összeset rátesszük az eseményre (klaszterméret pontozás miatt fontos!)
        all_cluster_ids = [n['id'] for n in items]
        
        if events:
            for ev in events:
                if isinstance(ev, dict):
                    ev['ids'] = all_cluster_ids
                    # Biztosítjuk a szótár-alapú elérést a későbbi szakaszokhoz
                    final_clusters.append(ev)
                else:
                    # Ha a modell véletlenül objektumot adna vissza
                    ev_dict = ev.dict() if hasattr(ev, 'dict') else vars(ev)
                    ev_dict['ids'] = all_cluster_ids
                    final_clusters.append(ev_dict)
            
        time.sleep(2.0) # Biztonságos várakozás a Lite hívások között a kvóta miatt

    return final_clusters, discarded_summaries

def auto_cluster(embeddings, news_pool):
    distance_limit = 0.125
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_limit,
        metric='cosine', 
        linkage='complete' 
    ).fit(embeddings)
    
    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])
        
    max_size = max(len(g) for g in groups.values()) if groups else 0
    myPrint(f"✨ Optimális klaszterezés elérve ({len(groups)} csoport, legnagyobb: {max_size} hír).")
    
    return groups
    
def filter_and_rank_clusters(clusters_data):
    """
    Végső szűrés a súlyozott pontszám alapján. 
    Csak a tényleg fontos hírek mennek tovább elemzésre.
    """
    final_selection = []
    
    for c in clusters_data:
        # Meghívjuk a korábban megírt pontozó függvényt
        # (Feltételezzük, hogy a c egy objektum vagy dict, amit a cluster_news ad vissza)
        scores = c.scores if hasattr(c, 'scores') else c.get('scores', {})
        total_score = calculate_priority_score(scores)
        
        # Csak az 5.0 feletti (vagy általad választott küszöb) hírek mennek tovább
        if total_score >= 5.0:
            # Eltároljuk a kerekített pontszámot a későbbi megjelenítéshez
            if isinstance(c, dict):
                c['total_score'] = round(total_score, 1)
            else:
                # Pydantic objektum esetén dinamikusan adjuk hozzá vagy kezeljük
                setattr(c, 'total_score', round(total_score, 1))
            
            final_selection.append(c)
    
    # Egy utolsó biztonsági sorbarendezés
    return sorted(final_selection, key=lambda x: getattr(x, 'total_score', 0) if not isinstance(x, dict) else x.get('total_score', 0), reverse=True)

import random
import time

def main():
    myPrint("🚀 Hírfigyelő rendszer indítása...")
    
    # 1. Lekérés
    raw_news = fetch_news()
    if not raw_news:
        myPrint("❌ Nincs bejövő hír, leállás.")
        return
    
    # --- Duplikátum szűrés ---
    seen_titles = set()
    unique_news = []
    for n in raw_news:
        clean_title = n['title'].strip().lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_news.append(n)
    
    myPrint(f"🧹 Duplikátumok eltávolítva: {len(raw_news)} -> {len(unique_news)} egyedi hír maradt.")
    
    # 2. Stratégiai témák generálása
    sample_size = min(len(unique_news), 300)
    titles_sample = "\n".join([n['title'] for n in random.sample(unique_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        myPrint("⚠️ Nem sikerült stratégiai témákat generálni, leállás.")
        return

    myPrint("🎯 Napi Stratégiai Topikok:")
    for i, t in enumerate(topics, 1):
        myPrint(f"  {i}. {t}")
        
    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés (Matek: Cosine távolság alapján)
    filtered_news = semantic_filter(unique_news, topics, top_p=0.86)
    if not filtered_news:
        myPrint("❌ A szemantikus szűrés után nem maradt hír, leállás.") 
        return

    # 4. Klaszterezés és Lite Validáció (Szintézis)
    # Ez a függvény most már megcsinálja a multilingvális beágyazást, 
    # a csoportosítást és a Lite alapú névadást/pontozást is.
    all_events, discarded_summaries = cluster_news(filtered_news)
    
    if not all_events:
        myPrint("❌ Nem sikerült eseményeket generálni a klaszterekből.")
        return

    # 5. Hibrid Pontozás és Sorbarendezés (LLM minőség + Klaszterméret)
    myPrint(f"⚖️ Végleges pontszámok kiszámítása {len(all_events)} eseményre...")
    for ev in all_events:
        ev['final_score'] = calculate_priority_score(ev)
    all_events.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    top_20_events = all_events[:20]

    # 6. Flash Elemzés (Mélyebb összefoglaló generálása)
    myPrint(f"🧠 Flash elemzés indítása a top {len(top_20_events)} eseményre...")
    final_data_package = []
    
    for i, event in enumerate(top_20_events, 1):
        merged_ids = event.get('ids', [])
        c_name = event.get('name', "Névtelen esemény")
        c_cat = event.get('category', "EGYÉB")
        c_score = event.get('final_score', 0)
        
        # Opcionális: A Lite 1 mondatos összefoglalóját is kinyerheted, ha beleteszed a promptba kontextusnak
        # c_summary_lite = getattr(event, 'summary', '') 
        
        # Összeszedjük a konkrét hír objektumokat az ID-k alapján
        relevant_news_objects = [n for n in filtered_news if n['id'] in merged_ids]
        
        if not relevant_news_objects:
            continue

        myPrint(f"  [{i}/{len(top_20_events)}] Összefoglalás: {c_name} (Pont: {c_score} | Cikkek: {len(relevant_news_objects)})")
        
        # Itt hívjuk a nagy Flash modellt
        summary = generate_event_summary(c_name, relevant_news_objects)
        
        sources_data = [
            {"name": n['source'], "url": n.get('link', '')} 
            for n in relevant_news_objects
        ]
        
        final_data_package.append({
            'category': c_cat,
            'title': c_name,
            'summary': summary,
            'sources': sources_data,
            'score': c_score
        })
        
        time.sleep(3) # Biztonsági szünet a Flash hívások között

    # 7. Kimenetek (HTML, Deploy, stb.)
    if final_data_package:
        myPrint("📦 Adatcsomag összeállítva, mentés és publikálás...")
        output_handler.process_and_send(final_data_package, topics_html, discarded_summaries)
        
    # 8. Statisztika
    from gemini_handler import usage_tracker        
    usage = usage_tracker.get_aggregated_stats()
    myPrint(f"📊 Token használat: {usage}")
    myPrint("✅ Kész.")
    
if __name__ == "__main__":
    main()
