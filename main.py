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

def calculate_priority_score(scores):
    """Kiszámítja a súlyozott összpontszámot a sorbarendezéshez."""
    # Kezeljük mind a Pydantic objektumot, mind a dict-et
    if hasattr(scores, 'model_dump'):
        s = scores.model_dump()
    elif isinstance(scores, dict):
        s = scores
    else:
        return 0
    
    return (s.get('relevance', 0) * 0.3) + \
           (s.get('impact', 0) * 0.5) + \
           (s.get('novelty', 0) * 0.2)
    
def semantic_filter(news_pool, topics, top_k=300):
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

    # 3. SZIGORÍTOTT ZAJ-KAPU: 0.3 helyett 0.65 (A sport és a fafajok itt hullanak ki)
    filtered = [n for n in news_pool if n.get('match_score', 0) > 0.65]
    filtered.sort(key=lambda x: x['match_score'], reverse=True)
    
    final_selection = filtered[:top_k]
    
    myPrint(f"✅ Rangsorolás kész: {len(final_selection)} hír továbbküldve (átlagos relevancia: {sum(n['match_score'] for n in final_selection)/len(final_selection) if final_selection else 0:.2f})")
    
    return final_selection

import time

def cluster_news(news_pool):
    if not news_pool: return []
    myPrint(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # 1. VEKTOROK ÚJRAHASZNOSÍTÁSA (Nincs API hívás, instant lefut!)
    embeddings = [n['embedding'] for n in news_pool]

    groups = auto_cluster(embeddings, news_pool)
    total_groups = len(groups)

    final_clusters = []
    i = 0
    for label, items in groups.items():
        i += 1
        
        # 2. TOKEN-SPÓROLÁS: Ha hatalmas a klaszter, csak az első 15 legrelevánsabb hírt mutatjuk meg a Lite-nak
        representative_items = items[:15]
        
        formatted_list = "\n".join([
            f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}" 
            for n in representative_items
        ])

        myPrint(f"  [{i}/{total_groups}] Lite validáció | Klaszter ID: {label} | {len(items)} hír (ebből {len(representative_items)} küldve)...")
        result = validate_news_clusters(formatted_list, topics=None) # A topics opcionális lehet itt
        
        events = []
        if isinstance(result, dict):
            events = result.get("events", [])
        elif hasattr(result, "events"):
            events = result.events

        # 3. AZ ID-K VISSZAÍRÁSA: A Lite csak 15 ID-t látott, de mi az összeset rátesszük az eseményre!
        all_cluster_ids = [n['id'] for n in items]
        
        if events:
            for ev in events:
                if isinstance(ev, dict):
                    ev['ids'] = all_cluster_ids
                else:
                    ev.ids = all_cluster_ids
            final_clusters.extend(events)
            
        time.sleep(3.0) # Biztonságos várakozás a Lite hívások között

    return final_clusters

from sklearn.cluster import AgglomerativeClustering

def auto_cluster(embeddings, news_pool):
    # A Cosine távolság 0 és 2 között mozog. 
    # A 0.15-ös távolság 85%-os koszinusz hasonlóságot jelent, ami tökéletes az azonos hírekhez.
    distance_limit = 0.15 
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_limit,
        metric='cosine', # EZ A LÉNYEG: Nyelvfüggetlen gömb-geometria
        linkage='average' # A cosine metrikához a 'ward' nem jó, az 'average' a stabil
    ).fit(embeddings)
    
    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])
        
    myPrint(f"✨ Optimális klaszterezés elérve ({len(groups)} masszív csoport).")
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

    raw_news = raw_news[:600]
    
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
    raw_news = unique_news[:100] # Ezzel dolgozunk tovább
    
    # 2. Stratégiai témák
    # Megjegyzés: random helyett az utolsó N hír is jó lehet, de a random segít a diverzitásban
    sample_size = min(len(raw_news), 300)
    titles_sample = "\n".join([n['title'] for n in random.sample(raw_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        myPrint("⚠️ Nem sikerült stratégiai témákat generálni.")
        return

    myPrint("TOP TOPIKOK:")
    for i, t in enumerate(topics, 1):
        myPrint(f"{i}: {t}")
        
    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés
    filtered_news = semantic_filter(raw_news, topics, top_k=300)
    if not filtered_news:
        myPrint("no semantic filtered news, exiting") 
        return

    # 4. Klaszterezés és szűrés
    all_events = cluster_news(filtered_news)
    initial_ranked = filter_and_rank_clusters(all_events)
    
    # --- ÚJ: Teljes lista logolása az elemzés előtt ---
    myPrint(f"📊 Összesen {len(initial_ranked)} releváns eseményt találtam:")
    for i, cluster in enumerate(initial_ranked, 1):
        name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen')
        score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)
        
        prefix = "✅ [TOP 20]" if i <= 20 else "❌ [KIMARAD]"
        myPrint(f"    {prefix} #{i} | {name} | Pontszám: {score}")
    # --------------------------------------------------

    # Stratégiai Szerkesztő fázis
    myPrint(f"🧠 Stratégiai felülvizsgálat és összevonás ({len(initial_ranked)} jelölt)...")
    refined_response = refine_event_list(initial_ranked[:20], topics)
    
    final_data_package = []
    refined_list = refined_response.get("refined_events", [])[:20]

    myPrint(refined_list)
    myPrint(f"🧠 Flash elemzés indítása {len(refined_list)} véglegesített eseményre...")
    
    for i, event in enumerate(refined_list, 1):
        # Összegyűjtjük az összes hírt az összes összevont ID-ból
        merged_ids = event.get("merged_ids", [])
        c_name = event.get("display_name", "Névtelen esemény")
        relevant_news_objects = [n for n in filtered_news if n['id'] in merged_ids]
        
        if not relevant_news_objects:
            continue

        myPrint(f"  [{i}/{len(refined_list)}] Összefoglalás: {c_name}")
        summary = generate_event_summary(c_name, relevant_news_objects)
        
        sources_data = [
            {"name": n['source'], "url": n.get('link', '')} 
            for n in relevant_news_objects
        ]
        
        final_data_package.append({
            'category': "EGYÉB", #c_cat,
            'title': c_name,
            'summary': summary,
            'sources': sources_data,
            'score': 0 #c_score
        })
        
        time.sleep(5) # Kicsit több szünet a biztonság kedvéért

    # 6. Kimenetek (ntfy, HTML, stb.)
    if final_data_package:
        #output_handler.process_and_send(final_data_package, topics_html)
        myPrint(final_data_package)
        
    # Statisztika
    from gemini_handler import usage_tracker        
    usage = usage_tracker.get_aggregated_stats()
    myPrint(f"📊 Token használat: {usage}")
    myPrint("✅ Kész.")
    
if __name__ == "__main__":
    main()
