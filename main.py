import config
import output_handler
from gemini_handler import get_strategic_topics, validate_news_clusters, generate_event_summary

import feedparser
import re
import telebot
import time
import json
from datetime import datetime

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
client = genai.Client(
    api_key=config.GOOGLE_API_KEY, 
    http_options={'api_version': 'v1beta'}
)
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def smart_truncate(text, max_length=600):
    if len(text) <= max_length: return text
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + "..."

def safe_generate_content(prompt, is_json_task=False, sys_instruct=None):
    target_model = config.MODEL_LITE_ID

    """Újrapróbálkozó függvény API limitek és szerverhibák kezelésére."""
    if is_json_task:
        #target_model = config.MODEL_ID
        current_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ClusterResult, # Fix sémakényszerítés
            system_instruction=sys_instruct
        )
    else:
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
    if is_json_task:
        return "{}" # Üres JSON objektum, hogy a json.loads ne omoljon össze
    return "Hiba: A tartalom generálása sikertelen."

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
        
        response = client.models.embed_content(
            model="gemini-embedding-001",
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
        n_clusters=None,
        distance_threshold=0.12,
        metric='cosine',
        linkage='complete' # Szigorúbb láncolás (average helyett)
    ).fit(embeddings)

    # A cluster_news függvényben a .fit(embeddings) után:
    #print(f"--- Klaszterezési statisztika ---")
    #print(f"Talált csoportok száma: {clustering.n_clusters_}")
    #print(f"Hírek besorolása (labels): {clustering.labels_}") # Megmutatja, melyik hír melyik sorszámú csoportba került

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    final_clusters = []
    sys_instruct = """Te egy tapasztalt hírszerkesztő vagy. 
    Feladatod: A megadott hírek közül válaszd ki azokat, amelyek ugyanarról az alapvető eseményről szólnak.
    SZABÁLYOK:
    1. Ha két hír ugyanarról a gazdasági bejelentésről vagy politikai eseményről szól, maradjanak egy csoportban, még ha más forrásból is vannak.
    2. A 'name' mező legyen egy rövid, tárgyilagos cím, SZIGORÚAN magyar nyelven! Kivéve cégnevek, azok maradjanak az eredeti formájukban.
    3. A 'category' (HAZAI/GLOBÁLIS/EGYÉB) besorolásnál a magyar vonatkozású híreket mindig jelöld HAZAI-nak. A világpolitika vagy gazdasági fontos hírek a GLOBÁLIS kategóriába mennek, a jelentéktelenebb külföldi hírek az EGYÉB-be.
    4. Pontozd az eseményt a megadott szempontok szerint."""

    i = 0
    for label, items in groups.items():
        # SZŰRÉS: Ha csak 1 hír van a matematikai csoportban, eldobjuk
        #if len(items) < 2:
        #    continue
            
        # A summary elejét is odaadjuk, hogy lássa a kontextust
        formatted_list = "\n".join([f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {n['summary'][:150]}..." for n in items])
        
        if i == 5:
            i = 0
            time.sleep(2)
            print("...")
        else:
            i = i + 1 
        ai_response = safe_generate_content(f"Hírek:\n{formatted_list}", True, sys_instruct)

        if ai_response and isinstance(ai_response, str): # Ellenőrizzük, hogy string-e
            try:
                data = json.loads(ai_response)
                print(f"DEBUG: Csoport neve: {data.get('name')} | Hírek száma: {len(data.get('ids', []))}")
                if data and data.get('ids'): 
                    final_clusters.append(data)
                else:
                    titles = [n['title'] for n in items]
                    titles_str = "\n - ".join(titles)
                    print(f"⚠️ AI elutasította a csoportot, mert ezek nem illenek össze:\n - {titles_str}")
            except:
                print(f"JSON hiba. Nyers válasz: {ai_response[:100]}")
    return final_clusters

def parse_clusters(clusters_data):
    filtered = []
    for c in clusters_data:
        s = c.get('scores', {})
        # 1. Alappontszám kiszámítása (1-10 skálán)
        base_score = (s.get('relevance', 0)*0.4) + (s.get('impact', 0)*0.4) + (s.get('novelty', 0)*0.2)
        
        # 2. Források számának lekérése az ID listából
        source_count = len(c.get('ids', []))
        
        # 3. Szorzás és kerekítés
        # Csak akkor számolunk tovább, ha az alappontszám eléri a küszöböt
        if base_score >= 5:
            # Itt szorzunk a források számával
            final_score = round(base_score * source_count, 1)
            c['total_score'] = final_score
            filtered.append(c)
            
    # A rendezés most már a felszorzott pontszám alapján történik
    return sorted(filtered, key=lambda x: x['total_score'], reverse=True)

def summarize_event(cluster_name, ids, news_pool):
    relevant = [n for n in news_pool if n['id'] in ids]
    source_count = len(relevant) # Megszámoljuk a forrásokat
    
    # Készítünk egy listát a források neveivel a promptba, hogy az AI lássa az elkülönítést
    text_parts = []
    for n in relevant:
        text_parts.append(f"[{n['source']}]: {n['title']} - {n['summary']}")
    
    input_text = "\n".join(text_parts)
    
    # Dinamikus utasítás a mondatszámra
    sys_instruct = f"""Te egy precíz hírszerkesztő vagy. 
    Írj pontosan {source_count} mondatos magyar összefoglalót! 
    Az összes forrást használd fel az elemzéshez, ha valamelyik irreleváns információt, érzelmi manipulációt vagy clickbaitet tartalmaz, azt említsd meg.
    Tilos a Markdown! Csak tiszta szöveg."""

    prompt = f"Esemény: {cluster_name}\n\nForrások száma: {source_count}\n\nHírek:\n{input_text}"
    
    response = safe_generate_content(prompt, sys_instruct=sys_instruct)
    return response.strip()

def main():
    news_pool = fetch_news()
    if not news_pool: return
    
    # 1. Klaszterezés és validáció (egyelőre marad sorban futó)
    clusters = parse_clusters(cluster_news(news_pool))
    if not clusters: return

    # 2. Az adatcsomag összeállítása az output_handler számára
    final_data_package = []
    
    print(f"Összefoglalók készítése {len(clusters)} csoporthoz...")
    for cluster in clusters:
        # Összefoglaló legenerálása (Lite)
        summary = summarize_event(cluster['name'], cluster['ids'], news_pool).strip()
        
        # Források kigyűjtése
        rel_news = [n for n in news_pool if n['id'] in cluster['ids']]
        sources_str = ", ".join(set([n['source'] for n in rel_news]))
        
        # Az elem hozzáadása a listához
        final_data_package.append({
            'category': cluster.get('category', 'EGYÉB'),
            'title': cluster['name'],
            'summary': summary,
            'sources': sources_str,
            'score': cluster.get('total_score', 0)
        })

    # 3. Átadás az output handlernek a küldéshez
    output_handler.process_and_send(final_data_package)

if __name__ == "__main__":
    main()
