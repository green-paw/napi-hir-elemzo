# saját importok
import config
import output_handler

# Csak a szükséges handler funkciók
from gemini_handler import (
    get_strategic_topics, validate_news_clusters, 
    generate_event_summary, get_gemini_embeddings, 
    translate_if_needed, ClusterResult
)

from rss_handler import fetch_news

# általános importok
import math
import time
from concurrent.futures import ThreadPoolExecutor # A gyors fordításhoz
from sklearn.cluster import AgglomerativeClustering

from datetime import datetime

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
    texts = [f"CÍM: {n['title']} KIVONAT: {n['summary'][:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.08, # Szigorúbb olló
        metric='cosine',
        linkage='complete'
    ).fit(embeddings)

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    final_clusters = []
    for label, items in groups.items():
        formatted_list = "\n".join([f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:200]}" for n in items])
        # A validate_news_clusters-ben az AI már a belső pontokat is nézheti
        data = validate_news_clusters(formatted_list, schema=ClusterResult)

        if data and data.get('ids'):
            final_clusters.append(data)
        time.sleep(0.5) 

    return final_clusters

def parse_clusters(clusters_data):
    filtered = []
    for c in clusters_data:
        s = c.get('scores', {})
        # Súlyozott pontszám
        base_score = (s.get('relevance', 0)*0.4) + (s.get('impact', 0)*0.4) + (s.get('novelty', 0)*0.2)
        if base_score >= 5:
            c['total_score'] = round(base_score, 1)
            filtered.append(c)
    return sorted(filtered, key=lambda x: x['total_score'], reverse=True)

def main():
    # 1. Lekérés
    raw_news = fetch_news()
    if not raw_news:
        myPrint("no raw news, exiting")
        return

    # 2. Stratégiai témák
    titles_sample = "\n".join([f"{n['title']}" for n in raw_news[:200]])
    myPrint(f"get_strategic_topics hívás, titles_sample: {len(titles_sample)} karakter")
    topics = get_strategic_topics(titles_sample)
    
    # 3. Szemantikus szűrés
    myPrint(f"semantic_filter hívás: raw_news: {len(raw_news)} elem, topics: {len(topics)} elem")
    filtered_news = semantic_filter(raw_news, topics)
    if not filtered_news:
        myPrint("no semantic filtered news, exiting") 
        return

    # 💡 SEBESSÉG OPTIMALIZÁLÁS: 
    filtered_news = sorted(filtered_news, key=lambda x: x.get('match_score', 0), reverse=True)[:200]

    # 4. Klaszterezés (már csak a szűrt híreken)
    clusters = parse_clusters(cluster_news(filtered_news))
    
    # 5. Összefoglalás és küldés
    final_data_package = []
    
    # 💡 LIMIT: Csak a top 15 legfontosabb eseményt elemezzük (sebesség + átláthatóság)
    top_clusters = clusters[:10] 

    myPrint(f"🧠 Elemzés indítása a top {len(top_clusters)} eseményre...")

    for cluster in top_clusters:
        relevant_news_objects = [n for n in filtered_news if n['id'] in cluster['ids']]
        
        if not relevant_news_objects:
            myPrint("no relevant_news_objects in cluster {cluster['name']}, skipping") 
            continue

        # 💡 ZAJSZŰRÉS: Csak akkor elemezzük, ha legalább 2 forrás ír róla
        if len(set(n['source'] for n in relevant_news_objects)) < 2:
            myPrint("relevant_news_objects < 2 in cluster {cluster['name']}, skipping") 
            continue

        summary = generate_event_summary(cluster['name'], relevant_news_objects)
        
        sources_data = [
            {"name": n['source'], "url": n.get('link', '')} 
            for n in relevant_news_objects
        ]
        
        final_data_package.append({
            'category': cluster.get('category', 'EGYÉB'),
            'title': cluster['name'],
            'summary': summary,
            'sources': sources_data,
            'score': cluster.get('total_score', 0)
        })
        
        # Rövid szünet a Gemini RPM limit miatt
        time.sleep(1)

    # 6. Kimenetek kezelése (HTML + Telegram)
    if final_data_package:
        output_handler.process_and_send(final_data_package)
    else:
        myPrint("⚠️ Nem találtam elemezhető híreseményt a szűrők alapján.")
        
    myPrint("✅ Kész.")

if __name__ == "__main__":
    main()
