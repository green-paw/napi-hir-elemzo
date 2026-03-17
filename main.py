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
    
def semantic_filter(news_pool, topics, top_k=300):
    if not topics or not news_pool: return news_pool
    myPrint(f"🔍 Szemantikus rangsorolás: {len(news_pool)} hír...")
    
    # 1. Témák és hírek vektorizálása
    topic_embs = get_gemini_embeddings(topics)
    # A cím és a tagek együtt jobb kontextust adnak az embeddingnek
    news_texts = [f"[{', '.join(n['tags'])}] {n['title']}" if n['tags'] else n['title'] for n in news_pool]
    news_embs = get_gemini_embeddings(news_texts)
    
    # 2. Relevancia számítása
    for i, n_emb in enumerate(news_embs):
        # Megnézzük, mennyire passzol a hír BÁRMELYIK témához
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        max_sim = max(sims) if sims else 0
        
        # Elmentjük a pontszámot a hír objektumba
        news_pool[i]['match_score'] = max_sim

    # 3. Sorbarendezés (legmagasabb pontszám elöl) és vágás
    # Csak azokat tartjuk meg, amiknek van egy minimális közük a témákhoz (pl. > 0.3), 
    # hogy a totál zajt (pl. sporthírek, ha nem kérted) kidobjuk.
    filtered = [n for n in news_pool if n.get('match_score', 0) > 0.3]
    filtered.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Kivesszük az első top_k darabot
    final_selection = filtered[:top_k]
    
    myPrint(f"✅ Rangsorolás kész: {len(final_selection)} hír továbbküldve (átlagos relevancia: {sum(n['match_score'] for n in final_selection)/len(final_selection) if final_selection else 0:.2f})")
    
    return final_selection

def cluster_news(news_pool):
    if not news_pool: return []
    myPrint(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # Szövegek előkészítése az embeddinghez
    texts = [f"CÍM: {n['title']} KIVONAT: {n['summary'][:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    groups = {}
    groups = auto_cluster(embeddings, news_pool, initial_threshold=1, max_cluster_size=20)
    total_groups = len(groups)

    final_clusters = []
    i = 0
    for label, items in groups.items():
        i += 1
        formatted_list = "\n".join([
            f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}" 
            for n in items
        ])

        myPrint(f"  [{i}/{total_groups}] Lite elemzés | Klaszter ID: {label} | {len(items)} hír...")
        result = validate_news_clusters(formatted_list)
        events = []

        # Pydantic vagy dict kezelés
        if isinstance(result, dict):
            events = result.get("events", [])
        elif hasattr(result, "events"):
            events = result.events

        if events:
            final_clusters.extend(events)
            
        # Rövid várakozás a kvóták miatt (most már klaszterenként egyszer)
        time.sleep(0.6)

    return final_clusters

def auto_cluster(embeddings, news_pool, initial_threshold=0.7, max_cluster_size=20):
    current_threshold = initial_threshold
    attempts = 0
    max_attempts = 20
    
    while attempts < max_attempts:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=current_threshold,
            metric='euclidean',
            linkage='ward'
        ).fit(embeddings)
        
        groups = {}
        for idx, label in enumerate(clustering.labels_):
            groups.setdefault(label, []).append(news_pool[idx])
        
        # Ellenőrizzük, van-e túl nagy csoport
        too_large = [len(items) for items in groups.values() if len(items) > max_cluster_size]
        
        if not too_large:
            myPrint(f"✨ Optimális klaszterezés elérve ({current_threshold:.2f} küszöbbel, {len(groups)} csoport).")
            return groups
        
        # Ha van túl nagy, szigorítunk (csökkentjük a küszöböt)
        myPrint(f"⚠️ Túl nagy csoportok ({max(too_large)} hír). Szigorítás: {current_threshold:.2f} -> {current_threshold - 0.1:.2f}")
        current_threshold -= 0.05
        attempts += 1
        
        # Biztonsági fék, ne menjen 0 alá
        if current_threshold < 0.1:
            break
            
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

def main():
    # 1. Lekérés
    raw_news = fetch_news()
    if not raw_news:
        myPrint("no raw news, exiting")
        return

    # --- ÚJ: Duplikátum szűrés ---
    seen_titles = set()
    unique_news = []
    for n in raw_news:
        # Tisztítjuk a címet (kisbetű, szóközök le) a pontosabb egyezésért
        clean_title = n['title'].strip().lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_news.append(n)
    
    myPrint(f"🧹 Duplikátumok kiszűrve: {len(raw_news)} -> {len(unique_news)} hír.")
    raw_news = unique_news # Ezzel dolgozunk tovább
    
    # 2. Stratégiai témák
    # Megjegyzés: random helyett az utolsó N hír is jó lehet, de a random segít a diverzitásban
    sample_size = min(len(raw_news), 300)
    titles_sample = "\n".join([n['title'] for n in random.sample(raw_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        myPrint("⚠️ Nem sikerült stratégiai témákat generálni.")
        return

    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés
    filtered_news = semantic_filter(raw_news, topics, top_k=300)
    if not filtered_news:
        myPrint("no semantic filtered news, exiting") 
        return

    # 4. Klaszterezés és szűrés
    # Itt a filter_and_rank_clusters-t használjuk (ami a korábbi parse_clusters javított verziója)
    all_events = cluster_news(filtered_news)
    top_clusters = filter_and_rank_clusters(all_events)

    if not top_clusters:
        myPrint("⚠️ Nem találtam magas pontszámú eseményt.")
        return

    # --- ÚJ: Teljes lista logolása az elemzés előtt ---
    myPrint(f"📊 Összesen {len(top_clusters)} releváns eseményt találtam:")
    for i, cluster in enumerate(top_clusters, 1):
        name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen')
        score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)
        
        prefix = "✅ [TOP 20]" if i <= 20 else "❌ [KIMARAD]"
        myPrint(f"    {prefix} #{i} | {name} | Pontszám: {score}")
    # --------------------------------------------------

    top_clusters = top_clusters[:20]
    
    # 5. Összefoglalás és küldés
    final_data_package = []
    myPrint(f"🧠 Elemzés indítása a top {len(top_clusters)} eseményre...")

    for i, cluster in enumerate(top_clusters, 1):
        # Pydantic vagy Dict kezelés biztonságosan
        c_ids = cluster.ids if hasattr(cluster, 'ids') else cluster.get('ids', [])
        c_name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen esemény')
        c_cat = cluster.category if hasattr(cluster, 'category') else cluster.get('category', 'EGYÉB')
        c_score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)

        relevant_news_objects = [n for n in filtered_news if n['id'] in c_ids]
        
        if not relevant_news_objects:
            continue

        # Látni fogod, épp melyik cikket írja
        myPrint(f"  [{i}/{total_top}] Összefoglalás: {c_name} (Súly: {c_score})")
        
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
