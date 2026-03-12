import config
import feedparser
import re
import telebot
import time
import json
from datetime import datetime
import pytz
from feedgen.feed import FeedGenerator

from google import genai
from google.genai import types, errors
from sklearn.cluster import AgglomerativeClustering
from pydantic import BaseModel, Field
from typing import List

# --- Sémák definiálása ---
class Scores(BaseModel):
    relevance: int = Field(description="Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából (1-10)")
    impact: int = Field(description="Az esemény súlya (1-10)")
    novelty: int = Field(description="Mennyire tartalmaz új információt (1-10)")

class ClusterResult(BaseModel):
    name: str = Field(description="Esemény neve és helyszíne")
    category: str = Field(description="Kategória: HAZAI, GLOBÁLIS vagy EGYÉB")
    scores: Scores
    ids: List[int] = Field(description="A csoportba ténylegesen beleillő hírek ID-jai")

# --- Konfiguráció inicializálása ---
client = genai.Client(api_key=config.GOOGLE_API_KEY)
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def smart_truncate(text, max_length=600):
    if len(text) <= max_length: return text
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + "..."

def safe_generate_content(prompt, is_json_task=False, sys_instruct=None):
    """Újrapróbálkozó függvény API limitek és szerverhibák kezelésére."""
    if is_json_task:
        target_model = config.MODEL_ID
        current_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ClusterResult, # Fix sémakényszerítés
            system_instruction=sys_instruct
        )
    else:
        target_model = config.MODEL_LITE_ID
        current_config = types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=sys_instruct
        )

    for attempt in range(5):
        try:
            time.sleep(2)
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=current_config
            )
            print(f"model: {target_model}, input tokens: {response.usage_metadata.prompt_token_count}, output tokens: {response.usage_metadata.candidates_token_count}")
            return response.text
        except errors.APIError as e:
            if any(x in str(e).lower() for x in ["429", "503", "quota"]):
                wait_time = (attempt + 1) * 10 
                print(f"Limit hiba, várás: {wait_time}s...")
                time.sleep(wait_time)
            else: raise e
    return "Hiba: A szerver tartósan túlterhelt."

def fetch_news():
    news_pool = []
    item_id = 0
    print("Hírek lekérése...")
    for name, url in config.RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                summary = entry.get('summary', entry.get('description', ''))
                clean_summary = smart_truncate(re.sub('<[^<]+?>', '', summary), 600)
                news_pool.append({
                    "id": item_id,
                    "source": name,
                    "title": entry.title,
                    "summary": clean_summary
                })
                item_id += 1
        except Exception as e: print(f"Hiba ({name}): {e}")
    return news_pool

def get_gemini_embeddings(texts):
    """Vektorok lekérése 100-as csomagokban (Batch limit kezelése)."""
    all_embeddings = []
    
    # 100-asával daraboljuk a listát
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        print(f"Embedding lekérése: {i+1} - {min(i+100, len(texts))} / {len(texts)}")
        
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        
        # A batch eredményeit hozzáadjuk a fő listához
        all_embeddings.extend([embedding.values for embedding in response.embeddings])
        
        # Rövid szünet a biztonság kedvéért (Rate limit védelem)
        if len(texts) > 100:
            time.sleep(1)
            
    return all_embeddings

def cluster_news(news_pool):
    if not news_pool: return []
    print("Hibrid klaszterezés...")
    texts = [f"CÍM: {n['title']} KIVONAT: {n['summary'][:200]}" for n in news_pool]
    embeddings = get_gemini_embeddings(texts)

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.35, metric='cosine', linkage='average'
    ).fit(embeddings)

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    final_clusters = []
    sys_instruct = """Te egy elit hírszerkesztő vagy. Csoportosíts és pontozz! 
    Csak az azonos eseményeket hagyd egy csoportban (Helyszín-elv!)."""

    for label, items in groups.items():
        formatted_list = "\n".join([f"ID:{n['id']} | CÍM: {n['title']}" for n in items])
        ai_response = safe_generate_content(f"Hírek:\n{formatted_list}", True, sys_instruct)
        try:
            data = json.loads(ai_response)
            if data.get('ids'): final_clusters.append(data)
        except: print("JSON hiba egy csoportnál.")
    return final_clusters

def parse_clusters(clusters_data):
    filtered = []
    for c in clusters_data:
        s = c.get('scores', {})
        score = (s.get('relevance', 0)*0.4) + (s.get('impact', 0)*0.4) + (s.get('novelty', 0)*0.2)
        c['total_score'] = round(score, 1)
        if score >= 6: filtered.append(c)
    return sorted(filtered, key=lambda x: x['total_score'], reverse=True)

def summarize_event(cluster_name, ids, news_pool):
    relevant = [n for n in news_pool if n['id'] in ids]
    sources = ", ".join(set([n['source'] for n in relevant]))
    text = "\n".join([f"{n['title']}: {n['summary']}" for n in relevant])
    
    sys_instruct = "Írj 5 mondatos magyar összefoglalót! Tilos a Markdown (vastagítás, dőlt betű)!"
    prompt = f"Esemény: {cluster_name}\n\nHírek:\n{text}"
    
    response = safe_generate_content(prompt, sys_instruct=sys_instruct)
    return f"{cluster_name.upper()}\n\n{response.strip()}\n\n(Forrás: {sources})"

def send_split_message(chat_id, text):
    MAX = 3900
    if len(text) <= MAX:
        bot.send_message(chat_id, f"🗞 AI HÍRELEMZÉS\n\n{text}")
        return
    # Egyszerűsített darabolás
    for i in range(0, len(text), MAX):
        bot.send_message(chat_id, text[i:i+MAX])

def main():
    news_pool = fetch_news()
    if not news_pool: return
    clusters = parse_clusters(cluster_news(news_pool))
    if not clusters: return

    final_reports = []
    for cat, title in [('HAZAI', 'MAGYARORSZÁG'), ('GLOBÁLIS', 'VILÁGHÍREK'), ('EGYÉB', 'EGYÉB')]:
        items = [c for c in clusters if c.get('category') == cat][:10]
        if items:
            final_reports.append(f"--- {title} ---")
            for item in items:
                final_reports.append(summarize_event(item['name'], item['ids'], news_pool))

    if final_reports:
        send_split_message(config.TELEGRAM_CHAT_ID, "\n\n".join(final_reports))

if __name__ == "__main__":
    main()
