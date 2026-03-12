import config
import feedparser
import re
import telebot
import time
import json
from datetime import datetime
import pytz
from feedgen.feed import FeedGenerator

# Új importok az SDK-hoz, a vektorizáláshoz és a sémákhoz
from google import genai
from google.genai import types, errors
from sklearn.cluster import AgglomerativeClustering
from pydantic import BaseModel, Field
from typing import List

# --- Sémák definiálása (Structured Outputs) ---
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

def safe_generate_content(prompt, is_json_task=False, schema=None):
    """Újrapróbálkozó függvény API limitek és szerverhibák kezelésére."""
    
    if is_json_task:
        target_model = config.MODEL_ID
        current_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema # Pydantic séma átadása
        )
    else:
        target_model = config.MODEL_LITE_ID
        current_config = types.GenerateContentConfig(
            temperature=0.1
        )

    for attempt in range(5):
        try:
            time.sleep(2) # Rate limit védelem az ingyenes szinthez
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=current_config
            )
            print(f"model: {target_model}, input tokens: {response.usage_metadata.prompt_token_count}, output tokens: {response.usage_metadata.candidates_token_count}")
            return response.text
        except errors.APIError as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "503" in error_msg or "quota" in error_msg:
                wait_time = (attempt + 1) * 10 
                print(f"API Limit/Túlterhelés. Várakozás {wait_time}mp... (Próbálkozás: {attempt+1}/5)")
                time.sleep(wait_time)
            else:
                raise e
    return "Hiba: A szerver tartósan túlterhelt."

def fetch_news():
    """Begyűjti a híreket az összes forrásból és egyedi ID-val látja el őket."""
    news_pool = []
    item_id = 0
    print("Hírek lekérése az RSS forrásokból...")
    
    for name, url in config.RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                summary = entry.get('summary', entry.get('description', ''))
                clean_summary = re.sub('<[^<]+?>', '', summary)[:600]
                
                news_pool.append({
                    "id": item_id,
                    "source": name,
                    "title": entry.title,
                    "summary": clean_summary
                })
                item_id += 1
        except Exception as e:
            print(f"Hiba a(z) {name} forrásnál: {e}")
            
    return news_pool

def get_gemini_embeddings(texts):
    """Vektorok lekérése a Gemini text-embedding-004 modellel."""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=texts,
        config=types.EmbedContentConfig(task_type="CLUSTERING")
    )
    return [embedding.values for embedding in response.embeddings]

def cluster_news(news_pool):
    """Hibrid klaszterezés: Embedding előszűrés + LLM validáció sémával."""
    if not news_pool:
        return []

    print("Vektorizálás...")
    texts_to_embed = [
        f"CÍM: {n['title']} KIVONAT: {n.get('summary', '')[:200].replace('\n', ' ')}" 
        for n in news_pool
    ]
    embeddings = get_gemini_embeddings(texts_to_embed)

    print("Matematikai csoportosítás...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.35,
        metric='cosine',
        linkage='average'
    ).fit(embeddings)

    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])

    final_clusters = []
    
    print(f"LLM validáció {len(groups)} csoporton...")
    for label, items in groups.items():
        formatted_list = ""
        for n in items:
            summary_slice = n['summary'][:200].replace('\n', ' ')
            formatted_list += f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {summary_slice}\n"

        # A prompt sokkal rövidebb, mert a Pydantic séma leírja a formátumot!
        prompt = f"""
        Te egy elit hírszerkesztő vagy. A feladatod a hírek csoportosítása és pontozása.
        
        SZABÁLYOK:
        1. Csak azokat a híreket tartsd meg a csoportban, amelyek TÉNYLEG ugyanarról az eseményről szólnak (Helyszín-elv!).
        2. Ha egy hír (ID) kilóg a csoportból, egyszerűen hagyd ki az 'ids' listából.
        
        Hírek:
        {formatted_list}
        """

        ai_response = safe_generate_content(prompt, is_json_task=True, schema=ClusterResult)
        
        try:
            # A válasz egy tiszta JSON string, egyből parse-olható
            cluster_data = json.loads(ai_response)
            if cluster_data and cluster_data.get('ids'):
                final_clusters.append(cluster_data)
        except Exception as e:
            print(f"Hiba a csoport feldolgozásánál: {e}")

    # Nincs több json.dumps()! Közvetlenül a Python listát adjuk vissza.
    return final_clusters

def parse_clusters(clusters_data):
    """Pontozás és szűrés a kész Python listán."""
    try:
        filtered = []
        # Itt már egy kész dict listát kapunk a json string helyett
        for c in clusters_data:
            s = c.get('scores', {})
            weighted_score = (s.get('relevance', 0) * 0.4) + \
                             (s.get('impact', 0) * 0.4) + \
                             (s.get('novelty', 0) * 0.2)
            
            c['total_score'] = round(weighted_score, 1)
            
            if weighted_score >= 6:
                filtered.append(c)
        
        filtered.sort(key=lambda x: x['total_score'], reverse=True)
        return filtered
    except Exception as e:
        print(f"Scoring Hiba: {e}")
        return []

def summarize_event(cluster_name, ids, news_pool):
    """Lite modelles összefoglaló generálás."""
    relevant_news = [n for n in news_pool if n['id'] in ids]
    sources_set = set([n['source'] for n in relevant_news])
    sources_str = ", ".join(sources_set)
    
    combined_text = "\n".join([f"{n['title']}: {n['summary']}" for n in relevant_news])
    
    prompt = f"""
    Az alábbi hírek ugyanarról az eseményről szólnak ({cluster_name}):
    {combined_text}

    Írj belőlük egyetlen, tárgyilagos, rövid (maximum 5 mondat), magyar nyelvű összefoglalót. Ha a források között ellentmondás van, emeld ki külön.
    Szigorúan tilos a Markdown formázás (vastagítás, csillagok, dőlt betű)! 
    """

    response = safe_generate_content(prompt)
    final_text = f"{cluster_name.upper()}\n\n{response.strip()}\n\n(Forrás: {sources_str})"
    return final_text

def send_split_message(chat_id, text):
    """Feldarabolja az üzenetet Telegramra."""
    MAX_CHARS = 3900
    if len(text) <= MAX_CHARS:
        bot.send_message(chat_id, f"🗞 AI HÍRELEMZÉS (1/1)\n\n{text}")
        return

    parts = []
    temp_text = text
    while temp_text:
        if len(temp_text) <= MAX_CHARS:
            parts.append(temp_text.strip())
            break
        
        split_index = temp_text.rfind('\n\n', 0, MAX_CHARS)
        if split_index == -1:
            split_index = temp_text.rfind('\n', 0, MAX_CHARS)
        if split_index == -1:
            split_index = MAX_CHARS
            
        parts.append(temp_text[:split_index].strip())
        temp_text = temp_text[split_index:].strip()

    total_parts = len(parts)
    for i, part in enumerate(parts, 1):
        header = f"🗞 AI HÍRELEMZÉS ({i}/{total_parts})\n\n"
        bot.send_message(chat_id, header + part)

def generate_rss_file(reports, filename="rss_output.xml"):
    """RSS feed generálása."""
    fg = FeedGenerator()
    fg.id('https://github.com/your-repo/ai-news-agent')
    fg.title('AI Hírelemző Összefoglaló')
    fg.author({'name': 'Gemini AI Agent'})
    fg.link(href='https://github.com/green-paw', rel='alternate')
    fg.language('hu')
    fg.description('Napi politikai és gazdasági összefoglalók több forrás alapján')

    for report in reports:
        lines = report.split('\n')
        title = lines[0].replace('📌', '').strip()
        content = "\n".join(lines[1:])

        fe = fg.add_entry()
        fe.id(f"{title}_{datetime.now().strftime('%Y%m%d_%H%M')}")
        fe.title(title)
        fe.description(content)
        fe.link(href='https://github.com/green-paw')
        fe.pubDate(datetime.now(pytz.utc))

    fg.rss_file(filename)
    print(f"RSS feed sikeresen elmentve: {filename}")

def main():
    print("Hírek gyűjtése...")
    news_pool = fetch_news()
    
    if not news_pool:
        print("Nem sikerült híreket letölteni.")
        return

    print(f"Összesen {len(news_pool)} hír beolvasva. Csoportosítás és pontozás...")
    
    # Közvetlenül megkapjuk a listát
    raw_clusters = cluster_news(news_pool)
    clusters = parse_clusters(raw_clusters)

    if not clusters:
        print("Nem születtek releváns hírcsoportok.")
        return

    hazai = [c for c in clusters if c.get('category') == 'HAZAI'][:10]
    globalis = [c for c in clusters if c.get('category') == 'GLOBÁLIS'][:10]
    egyeb = [c for c in clusters if c.get('category') == 'EGYÉB'][:10]

    final_reports = []
    
    def process_section(section_list, section_title):
        if section_list:
            final_reports.append(f"--- {section_title} ({len(section_list)} esemény) ---")
            for item in section_list:
                report = summarize_event(item['name'], item['ids'], news_pool)
                final_reports.append(report)

    print("Összefoglalók készítése szekciónként...")
    process_section(hazai, "MAGYARORSZÁG ÉS RELEVÁNS HÍREK")
    process_section(globalis, "KIEMELT GLOBÁLIS ESEMÉNYEK")
    process_section(egyeb, "EGYÉB FONTOS HÍREK A VILÁGBÓL")

    reports_count = len(clusters)
    
    if len(final_reports) > 0:
        print(f"Kész! {reports_count} releváns esemény összefoglalva.")
        full_message = "\n\n".join(final_reports)
        
        try:
            print("Küldés Telegramra...")
            send_split_message(config.TELEGRAM_CHAT_ID, full_message)
        except Exception as e:
            print(f"Telegram hiba: {e}")

        try:
            generate_rss_file(final_reports, "rss_output.xml")
        except Exception as e:
            print(f"RSS hiba: {e}")
    else:
        print("A szűrési feltételeknek (Score >= 6) egyetlen hír sem felelt meg.")
        
if __name__ == "__main__":
    main()
