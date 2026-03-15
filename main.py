import config
import output_handler
from gemini_handler import get_strategic_topics, validate_news_clusters, generate_event_summary, get_gemini_embeddings, translate_if_needed

import feedparser
import re
import telebot
import json
import math
import time
from datetime import datetime, timedelta
import html

from google import genai
from google.genai import types, errors
from sklearn.cluster import AgglomerativeClustering
from pydantic import BaseModel, Field
from typing import List

# --- Sémák definiálása (Ezt a gemini_handler is látni fogja) ---
class Scores(BaseModel):
    relevance: int = Field(description="Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából (1-10)")
    impact: int = Field(description="Az esemény súlya (1-10)")
    novelty: int = Field(description="Mennyire tartalmaz új információt (1-10)")

class ClusterResult(BaseModel):
    name: str = Field(description="Esemény neve és helyszíne")
    category: str = Field(description="Kategória: HAZAI, GLOBÁLIS vagy EGYÉB")
    scores: Scores
    ids: List[int] = Field(description="A csoportba ténylegesen beleillő hírek ID-jai")

bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def clean_news_text(entry, field='title'):
    raw = entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))
    if not raw: return ""
    clean = re.sub(r'<[^>]+?>', '', html.unescape(raw))
    return " ".join(clean.split()).strip()

def smart_truncate(text, max_length=600):
    if len(text) <= max_length: return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def fetch_news():
    news_pool = []
    item_id = 0
    now = datetime.now()
    limit = timedelta(hours=48)
    # A diagnosztika alapján összeállított feketelista
    BLACKLIST = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept", "életmód", "bulvar"]

    print(f"📰 Hírek lekérése és szűrése ({limit.days * 24}h limit)...")
    
    for name, url in config.RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # 1. Dátumszűrés (published_parsed használatával)
                dt = now # Fallback, ha nincs dátum
                if hasattr(entry, 'published_parsed'):
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if now - dt > limit:
                        continue
                
                # 2. Kategória szűrés (tags alapján)
                tags = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
                if any(bad in tags for bad in BLACKLIST):
                    continue

                # 3. Tisztítás és mentés
                title = clean_news_text(entry, 'title')
                if not title: continue

                news_pool.append({
                    "id": item_id,
                    "source": name,
                    "title": title,
                    "summary": smart_truncate(clean_news_text(entry, 'summary'), 600),
                    "tags": tags,
                    "published": dt
                })
                item_id += 1
        except Exception as e: 
            print(f"⚠️ Hiba ({name}): {e}")
            
    print(f"✅ Kész! {len(news_pool)} releváns hír begyűjtve.")
    return news_pool

# --- ÚJ: Szemantikus szűrő matematikai alapjai ---
def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if not magnitude1 or not magnitude2: return 0
    return dot_product / (magnitude1 * magnitude2)

def semantic_filter(news_pool, topics):
    if not topics or not news_pool: 
        return news_pool

    print(f"🔍 Szemantikus szűrés indítása {len(news_pool)} híren...")
    
    topic_embeddings = get_gemini_embeddings(topics)
    news_texts = [n['title'] for n in news_pool] 
    news_embeddings = get_gemini_embeddings(news_texts)
    filtered_news = []
    threshold = 0.88 

    for i, n_emb in enumerate(news_embeddings):
        similarities = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embeddings]
        if not similarities: continue
            
        max_sim = max(similarities)
        
        # Ez a rész maradhat a debughoz, amíg be nem állítod a tökéletes értéket
        if max_sim >= threshold:
            news_pool[i]['match_score'] = round(max_sim, 2)
            filtered_news.append(news_pool[i])

    print(f"🎯 Szűrés kész. Maradt: {len(filtered_news)} / {len(news_pool)} hír.")
    return filtered_news

def cluster_news(news_pool):
    if not news_pool: return []
    print("🧩 Hibrid klaszterezés...")
    texts = [f"CÍM: {n['title']} KIVONAT: {n['summary'][:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.08,
        metric='cosine',
        linkage='complete'
    ).fit(embeddings)

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    final_clusters = []
    for label, items in groups.items():
        formatted_list = "\n".join([f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}..." for n in items])
        
        # --- ÚJ: Itt hívjuk a gemini_handler tiszta függvényét! ---
        data = validate_news_clusters(formatted_list, schema=ClusterResult)

        if data and data.get('ids'):
            print(f"✔️ Csoport elfogadva: {data.get('name')} ({len(data.get('ids'))} hír)")
            final_clusters.append(data)
        else:
            print(f"⚠️ AI elutasította a csoportot (Zaj vagy szétesett JSON).")
            
        time.sleep(1) # Kicsi szünet a rate limit miatt a cikluson belül

    return final_clusters

def parse_clusters(clusters_data):
    filtered = []
    for c in clusters_data:
        s = c.get('scores', {})
        base_score = (s.get('relevance', 0)*0.4) + (s.get('impact', 0)*0.4) + (s.get('novelty', 0)*0.2)
        source_count = len(c.get('ids', []))
        
        if base_score >= 5:
            c['total_score'] = round(base_score, 1)
            filtered.append(c)
            
    return sorted(filtered, key=lambda x: x['total_score'], reverse=True)

def summarize_event(cluster_name, ids, news_pool):
    relevant = [n for n in news_pool if n['id'] in ids]
    text_parts = [f"[{n['source']}]: {n['title']} - {n['summary']}" for n in relevant]
    input_text = "\n".join(text_parts)
    
    # --- ÚJ: Letisztult hívás a gemini_handler felé (Mondatszám nélkül) ---
    summary = generate_event_summary(cluster_name, input_text)
    return summary

def deep_inspect_rss():
    for name, url in config.RSS_SOURCES.items():
        print(f"=== FORRÁS: {name} ===")
        feed = feedparser.parse(url)
        if feed.entries:
            entry = feed.entries[0]
            
            # Kilistázzuk az összes elérhető mezőt
            fields = list(entry.keys())
            print(f"Elérhető mezők: {fields}")
            
            # Ha van benne valami izgalmas, nézzünk bele
            if 'tags' in fields:
                print(f"Példa tag: {entry.tags[0].term if entry.tags else 'Nincs term'}")
            if 'category' in fields:
                print(f"Kategória: {entry.category}")
        else:
            print("Üres feed.")
        print("-" * 30)

def main():
    # 1. Lekérés
    raw_news = fetch_news()
    if not raw_news: return

    for item in raw_news:
        item['title'] = translate_if_needed(item['title'])
    
    # 2. Stratégiai témák kinyerése (AI Flash motor)
    titles_only = "\n".join([f"{i+1}. {n['title']}" for i, n in enumerate(raw_news)])
    topics = get_strategic_topics(titles_only)
    print(f"💡 Napi fő témák: {', '.join(topics) if topics else 'Nem talált egyértelmű mintát.'}")

    # 3. Szemantikus szűrés (Matematika)
    filtered_news = semantic_filter(raw_news, topics)
    if not filtered_news: return

    # 4. Klaszterezés és AI validáció
    clusters = parse_clusters(cluster_news(filtered_news))
    if not clusters: return

    # 5. Összefoglalók legenerálása és adatcsomag összeállítása
    final_data_package = []
    print(f"✍️ Összefoglalók készítése {len(clusters)} csoporthoz...")
    
    for cluster in clusters:
        summary = summarize_event(cluster['name'], cluster['ids'], filtered_news)
        rel_news = [n for n in filtered_news if n['id'] in cluster['ids']]
        sources_str = ", ".join(set([n['source'] for n in rel_news]))
        
        final_data_package.append({
            'category': cluster.get('category', 'EGYÉB'),
            'title': cluster['name'],
            'summary': summary,
            'sources': sources_str,
            'score': cluster.get('total_score', 0)
        })

    # 6. Küldés Telegramra
    output_handler.process_and_send(final_data_package)
    print("✅ Folyamat befejezve.")

if __name__ == "__main__":
    main()
