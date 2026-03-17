# saját importok
import config
import output_handler

# Csak a szükséges handler funkciók
from gemini_handler import (
    get_strategic_topics, validate_news_clusters, 
    generate_event_summary, get_gemini_embeddings, 
    translate_if_needed, MultiClusterResponse
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

def calculate_priority_score(scores):
    """Kiszámítja a súlyozott összpontszámot a sorbarendezéshez."""
    # Kezeljük mind a Pydantic objektumot, mind a dict-et
    if hasattr(scores, 'model_dump'):
        s = scores.model_dump()
    elif isinstance(scores, dict):
        s = scores
    else:
        return 0
    
    # A te képleted: 40% relevancia, 40% hatás, 20% újdonság
    return (s.get('relevance', 0) * 0.4) + \
           (s.get('impact', 0) * 0.4) + \
           (s.get('novelate', 0) * 0.2) # Figyelem: novelty-re javítva, ha a sémádban novelate van
    
def semantic_filter(news_pool, topics):
    if not topics or not news_pool: return news_pool
    myPrint(f"🔍 Szemantikus szűrés: {len(news_pool)} hír...")
    
    # 1. Témák és hírek vektorizálása
    topic_embs = get_gemini_embeddings(topics)
    news_texts = [f"[{', '.join(n['tags'])}] {n['title']}" if n['tags'] else n['title'] for n in news_pool]
    news_embs = get_gemini_embeddings(news_texts)
    
    filtered = []
    threshold = 0.72

    # 2. Összehasonlítás
    for i, n_emb in enumerate(news_embs):
        # Kiszámoljuk a hasonlóságot a hír és az ÖSSZES téma között
        # Ez egy listát ad vissza (pl. [0.21, 0.45, 0.12, 0.33...])
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        
        # Kiválasztjuk a legmagasabb pontszámot (melyik témához áll a legközelebb?)
        max_sim = max(sims) if sims else 0

        # DEBUG: Csak az első pár hírnél nézzük meg a számokat
        if i < 5:
            myPrint(f"DEBUG: '{news_pool[i]['title'][:30]}...' max_sim: {max_sim}")
            
        # 3. Szűrés a küszöb alapján
        if max_sim >= threshold:
            news_pool[i]['match_score'] = round(max_sim, 2)
            filtered.append(news_pool[i])
    
    myPrint(f"✅ Szűrés kész: {len(filtered)} hír maradt (küszöb: {threshold})")
    return filtered

def cluster_news(news_pool):
    if not news_pool: return []
    myPrint(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # Szövegek előkészítése az embeddinghez
    texts = [f"CÍM: {n['title']} KIVONAT: {n['summary'][:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    # Laza matematikai csoportosítás
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.15, 
        metric='cosine',
        linkage='average'
    ).fit(embeddings)

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    print(f"matematikai csoprtosítás eredménye: {groups}")

    return []
    
    final_clusters = []
    for label, items in groups.items():
        # Ha a kupac túl nagy, szeleteljük fel fix 20-as darabokra
        # Így garantáltan nem kapunk 300 soros JSON-t
        chunks = [items[i:i + 20] for i in range(0, len(items), 20)]
        
        for chunk in chunks:
            formatted_list = "\n".join([
                f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}" 
                for n in chunk
            ])
            
            result = validate_news_clusters(formatted_list)
            events = []

            # Ellenőrizzük, hogy kaptunk-e eseményeket (lehet dict vagy Pydantic objektum)
            if isinstance(result, dict):
                events = result.get("events", [])
            elif hasattr(result, "events"):
                events = result.events
    
            if events:
                final_clusters.extend(events)
            
        # Rövid várakozás a kvóták miatt
        time.sleep(0.5)

    return final_clusters

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

def main():
    # 1. Lekérés
    raw_news = fetch_news()
    if not raw_news:
        myPrint("no raw news, exiting")
        return

    # 2. Stratégiai témák
    # Megjegyzés: random helyett az utolsó N hír is jó lehet, de a random segít a diverzitásban
    sample_size = min(len(raw_news), 200)
    titles_sample = "\n".join([n['title'] for n in random.sample(raw_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        myPrint("⚠️ Nem sikerült stratégiai témákat generálni.")
        return

    myPrint(f"🎯 Azonosított témák: {', '.join(topics)}")
    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés
    filtered_news = semantic_filter(raw_news, topics)
    if not filtered_news:
        myPrint("no semantic filtered news, exiting") 
        return

    # Sorbarendezés match_score alapján (ha a semantic_filter ad ilyet)
    filtered_news = sorted(filtered_news, key=lambda x: x.get('match_score', 0), reverse=True)[:300]

    # 4. Klaszterezés és szűrés
    # Itt a filter_and_rank_clusters-t használjuk (ami a korábbi parse_clusters javított verziója)
    all_events = cluster_news(filtered_news)
    top_clusters = filter_and_rank_clusters(all_events)[:20] 

    if not top_clusters:
        myPrint("⚠️ Nem találtam elég magas pontszámú eseményt.")
        return

    # 5. Összefoglalás és küldés
    final_data_package = []
    myPrint(f"🧠 Elemzés indítása a top {len(top_clusters)} eseményre...")

    for cluster in top_clusters:
        # Pydantic vagy Dict kezelés biztonságosan
        c_ids = cluster.ids if hasattr(cluster, 'ids') else cluster.get('ids', [])
        c_name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen esemény')
        c_cat = cluster.category if hasattr(cluster, 'category') else cluster.get('category', 'EGYÉB')
        c_score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)

        relevant_news_objects = [n for n in filtered_news if n['id'] in c_ids]
        
        if not relevant_news_objects:
            continue

        # A Flash modell hívása az elemzéshez
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
        
        time.sleep(1.2) # Kicsit több szünet a biztonság kedvéért

    # 6. Kimenetek (ntfy, HTML, stb.)
    if final_data_package:
        output_handler.process_and_send(final_data_package, topics_html)
        
    # Statisztika
    from gemini_handler import usage_tracker        
    usage = usage_tracker.get_aggregated_stats()
    myPrint(f"📊 Token használat: {usage}")
    myPrint("✅ Kész.")
    
if __name__ == "__main__":
    main()
