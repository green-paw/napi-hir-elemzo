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

# --- Szemantikus szűrő matematikai alapjai ---
def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0

def semantic_filter(news_pool, topics):
    if not topics or not news_pool: return news_pool
    print(f"🔍 Szemantikus szűrés: {len(news_pool)} hír...")
    
    topic_embs = get_gemini_embeddings(topics)
    # Itt használjuk ki a tags-eket is a pontossághoz!
    news_texts = [f"[{', '.join(n['tags'])}] {n['title']}" if n['tags'] else n['title'] for n in news_pool]
    news_embs = get_gemini_embeddings(news_texts)
    
    filtered = []
    threshold = 0.88 

    for i, n_emb in enumerate(news_embs):
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        max_sim = max(sims) if sims else 0
        if max_sim >= threshold:
            news_pool[i]['match_score'] = round(max_sim, 2)
            filtered.append(news_pool[i])
    return filtered

def cluster_news(news_pool):
    if not news_pool: return []
    print(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
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
        formatted_list = "\n".join([f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}" for n in items])
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
    # 1. Lekérés (rss_handler-ből, már szűrve dátumra és kategóriára)
    raw_news = fetch_news()
    if not raw_news: return

    # 2. GYORSÍTÁS: Párhuzamos fordítás
    print(f"🌍 Fordítás indítása {len(raw_news)} hírre...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        titles = [n['title'] for n in raw_news]
        translated_titles = list(executor.map(translate_if_needed, titles))
        for i, translated in enumerate(translated_titles):
            raw_news[i]['title'] = translated

    # 3. Stratégiai témák (Csak a Top 7-et kérjük)
    titles_only = "\n".join([f"{i+1}. {n['title']}" for i, n in enumerate(raw_news)])
    topics = get_strategic_topics(titles_only)
    
    # 4. Szűrés és Klaszterezés
    filtered_news = semantic_filter(raw_news, topics)
    if not filtered_news: return

    clusters = parse_clusters(cluster_news(filtered_news))
    
    # 5. Összefoglalás és küldés
    final_data_package = []
    for cluster in clusters:
        relevant = [n for n in filtered_news if n['id'] in cluster['ids']]
        input_text = "\n".join([f"[{n['source']}]: {n['title']} - {n['summary']}" for n in relevant])
        
        summary = generate_event_summary(cluster['name'], input_text)
        sources_data = [
            {"name": n['source'], "url": n.get('link', '')} 
            for n in relevant
        ]
        
        final_data_package.append({
            'category': cluster.get('category', 'EGYÉB'),
            'title': cluster['name'],
            'summary': summary,
            'sources': sources_data,
            'score': cluster.get('total_score', 0)
        })

    output_handler.process_and_send(final_data_package)
    print("✅ Kész.")

if __name__ == "__main__":
    main()
