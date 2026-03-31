from typing import List
from models import Article, ClusterResultSingle
import output_handler

# Csak a szükséges handler funkciók
from gemini_handler import (
    generate_structured_summary, get_strategic_topics, validate_news_clusters, 
    get_gemini_embeddings
)

from source import fetch_news

# általános importok
import math
import time
from sklearn.cluster import AgglomerativeClustering

from datetime import datetime
import random

import builtins
from datetime import datetime

_original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    _original_print(f"{timestamp} ", *args, **kwargs)

builtins.print = timestamped_print

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
    
def semantic_filter(news_pool: List[Article], topics: List[str], top_k=300):
    if not topics or not news_pool: return news_pool
    print(f"🔍 Szemantikus rangsorolás: {len(news_pool)} hír...")
    
    # 1. Témák és hírek vektorizálása
    topic_embs = get_gemini_embeddings(topics)
    # A cím és a tagek együtt jobb kontextust adnak az embeddingnek
    news_texts = [f"[{', '.join(n.tags)}] {n.title}" if n.tags else n.title for n in news_pool]
    news_embs = get_gemini_embeddings(news_texts)
    
    # 2. Relevancia számítása
    for i, n_emb in enumerate(news_embs):
        # Megnézzük, mennyire passzol a hír BÁRMELYIK témához
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        max_sim = max(sims) if sims else 0
        
        # Elmentjük a pontszámot a hír objektumba
        news_pool[i].match_score = max_sim

    # 3. Sorbarendezés (legmagasabb pontszám elöl) és vágás
    # Csak azokat tartjuk meg, amiknek van egy minimális közük a témákhoz (pl. > 0.3), 
    # hogy a totál zajt (pl. sporthírek, ha nem kérted) kidobjuk.
    filtered = [n for n in news_pool if n.match_score > 0.3]
    filtered.sort(key=lambda x: x.match_score, reverse=True)
    
    # Kivesszük az első top_k darabot
    final_selection = filtered[:top_k]
    
    print(f"✅ Rangsorolás kész: {len(final_selection)} hír továbbküldve (átlagos relevancia: {sum(n.match_score for n in final_selection)/len(final_selection) if final_selection else 0:.2f})")
    
    return final_selection

def cluster_news(news_pool: List[Article]) -> List[ClusterResultSingle]:
    if not news_pool: return []
    print(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # Szövegek előkészítése az embeddinghez
    texts = [f"CÍM: {n.title} KIVONAT: {n.summary[:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    groups = {}
    groups = auto_cluster(embeddings, news_pool, initial_threshold=1, max_cluster_size=20)
    total_groups = len(groups)

    final_clusters = []
    i = 0
    for label, items in groups.items():
        i += 1
        formatted_list = "\n".join([
            f"ID:{n.id} | CÍM: {n.title} | KIVONAT: {n.summary[:150]}" 
            for n in items
        ])

        print(f"  [{i}/{total_groups}] Lite elemzés | Klaszter ID: {label} | {len(items)} hír...")
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

def auto_cluster(embeddings: List[List[float]], news_pool: List[Article], initial_threshold=0.7, max_cluster_size=20) -> dict:
    current_threshold = initial_threshold
    attempts = 0
    max_attempts = 20
    
    groups = {}
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
            print(f"✨ Optimális klaszterezés elérve ({current_threshold:.2f} küszöbbel, {len(groups)} csoport).")
            return groups
        
        # Ha van túl nagy, szigorítunk (csökkentjük a küszöböt)
        print(f"⚠️ Túl nagy csoportok ({max(too_large)} hír). Szigorítás: {current_threshold:.2f} -> {current_threshold - 0.05:.2f}")
        current_threshold -= 0.05
        attempts += 1
        
        # Biztonsági fék, ne menjen 0 alá
        if current_threshold < 0.1:
            break
            
    return groups

def filter_and_rank_clusters(clusters_data: List[ClusterResultSingle]) -> List[ClusterResultSingle]:
    """
    Végső szűrés a súlyozott pontszám alapján. 
    Csak a tényleg fontos hírek mennek tovább elemzésre.
    """
    final_selection: List[ClusterResultSingle] = []
    
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
        print("no raw news, exiting")
        return
    
    # 2. Stratégiai témák
    # Megjegyzés: random helyett az utolsó N hír is jó lehet, de a random segít a diverzitásban
    sample_size = min(len(raw_news), 300)
    titles_sample = "\n".join([n.title for n in random.sample(raw_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        print("⚠️ Nem sikerült stratégiai témákat generálni.")
        return

    print("TOP TOPIKOK:")
    for i, t in enumerate(topics, 1):
        print(f"{i}: {t}")
        
    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés
    filtered_news = semantic_filter(raw_news, topics, top_k=300)
    if not filtered_news:
        print("no semantic filtered news, exiting") 
        return

    # 4. Klaszterezés és szűrés
    # Itt a filter_and_rank_clusters-t használjuk (ami a korábbi parse_clusters javított verziója)
    all_events = cluster_news(filtered_news)
    top_clusters = filter_and_rank_clusters(all_events)

    if not top_clusters:
        print("⚠️ Nem találtam magas pontszámú eseményt.")
        return

    # --- ÚJ: Teljes lista logolása az elemzés előtt ---
    print(f"📊 Összesen {len(top_clusters)} releváns eseményt találtam:")
    for i, cluster in enumerate(top_clusters, 1):
        name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen')
        score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)
        
        prefix = "✅ [TOP 30]" if i <= 30 else "❌ [KIMARAD]"
        print(f"    {prefix} #{i} | {name} | Pontszám: {score}")
    # --------------------------------------------------

    top_clusters = top_clusters[:30]
    total_top = len(top_clusters)
    
    # 5. Összefoglalás és küldés
    final_data_package = []
    print(f"🧠 Elemzés indítása a top {total_top} eseményre...")

    for i, cluster in enumerate(top_clusters, 1):
        # Pydantic vagy Dict kezelés biztonságosan
        c_ids = cluster.ids if hasattr(cluster, 'ids') else cluster.get('ids', [])
        c_name = cluster.name if hasattr(cluster, 'name') else cluster.get('name', 'Névtelen esemény')
        c_cat = cluster.category if hasattr(cluster, 'category') else cluster.get('category', 'EGYÉB')
        c_score = getattr(cluster, 'total_score', 0) if not isinstance(cluster, dict) else cluster.get('total_score', 0)

        relevant_news_objects = [n for n in filtered_news if n.id in c_ids]
        
        if not relevant_news_objects:
            continue

        # Látni fogod, épp melyik cikket írja
        print(f"  [{i}/{total_top}] Összefoglalás: {c_name} (Súly: {c_score})")
        
        # 1. Lekérjük a strukturált szótárat az LLM-től
        summary_data = generate_structured_summary(c_name, relevant_news_objects)
        
        # 2. Összeállítjuk a Markdown formátumot a Telegram és a HTML számára
        formatted_summary = f"{summary_data.get('summary', 'Hiányzó összefoglaló.')}\n\n"
        
        # Baloldali narratíva kezelése
        left_analysis = summary_data.get('left_wing_analysis', '').strip()
        if left_analysis:
            formatted_summary += f"**Baloldali / Liberális narratíva:**\n{left_analysis}\n\n"
        else:
            formatted_summary += f"**Baloldali / Liberális narratíva:**\n_Nincs releváns forrás ehhez az oldalhoz._\n\n"
            
        # Jobboldali narratíva kezelése
        right_analysis = summary_data.get('right_wing_analysis', '').strip()
        if right_analysis:
            formatted_summary += f"**Jobboldali / Konzervatív narratíva:**\n{right_analysis}\n\n"
        else:
            formatted_summary += f"**Jobboldali / Konzervatív narratíva:**\n_Nincs releváns forrás ehhez az oldalhoz._\n\n"

        # 3. Források kigyűjtése
        sources_data = [
            {"name": n.source, "url": n.link or ''} 
            for n in relevant_news_objects
        ]
        
        # 4. Csomagolás a kimenethez (itt a 'summary' kulcs alá már a formázott szöveget tesszük)
        final_data_package.append({
            'category': c_cat,
            'title': c_name,
            'summary': formatted_summary.strip(), 
            'sources': sources_data,
            'score': c_score
        })
        
        #time.sleep(1.2) # Kicsit több szünet a biztonság kedvéért

    # 6. Kimenetek (ntfy, HTML, stb.)
    if final_data_package:
        output_handler.process_and_send(final_data_package)
        
    # Statisztika
    try:
        from llm_core import usage_tracker        
        usage = usage_tracker.get_aggregated_stats()
        print(f"📊 Token használat: {usage}")
    except:
        pass

    print("✅ Kész.")
    
if __name__ == "__main__":
    main()
