import config
import feedparser
import re
import telebot
from google import genai
import time
from google.genai import errors
from feedgen.feed import FeedGenerator
from datetime import datetime
import pytz
import json
from google.genai import types
from sklearn.cluster import AgglomerativeClustering

# --- Konfiguráció inicializálása ---
client = genai.Client(api_key=config.GOOGLE_API_KEY)
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def safe_generate_content(prompt, is_json_task=False):
    """Újrapróbálkozó függvény API limitek és szerverhibák kezelésére."""
    
    # 1. Állapot: Klaszterezés és pontozás (Precíziós feladat)
    if is_json_task:
        target_model = config.MODEL_ID
        current_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
    # 2. Állapot: Összefoglaló írása (Kreatív/szöveges feladat)
    else:
        target_model = config.MODEL_LITE_ID
        current_config = types.GenerateContentConfig(
            temperature=0.1
        )

    for attempt in range(3): # Max 3 próbálkozás
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=current_config
            )
            print(f"model: {target_model}, input tokens: {response.usage_metadata.prompt_token_count}, output tokens: {response.usage_metadata.candidates_token_count}")
            return response.text
        except errors.ServerError as e:
            if "503" in str(e) or "high demand" in str(e):
                print(f"Szerver túlterhelt, várakozás... (Próbálkozás: {attempt+1}/3)")
                time.sleep(5) # Vár 5 másodpercet az újabb próbálkozás előtt
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
            # Forrásonként az utolsó 10 hírt vesszük figyelembe
            for entry in feed.entries[:10]:
                summary = entry.get('summary', entry.get('description', ''))
                # Tisztítás: HTML tagek eltávolítása és hossz korlátozása
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
    """Hibrid klaszterezés: Embedding előszűrés + LLM validáció."""
    if not news_pool:
        return "[]"

    print("Vektorizálás...")
    # 1. Szövegek előkészítése (Cím + Rövidített kivonat)
    texts_to_embed = [
        f"CÍM: {n['title']} KIVONAT: {n.get('summary', '')[:200].replace('\n', ' ')}" 
        for n in news_pool
    ]
    embeddings = get_gemini_embeddings(texts_to_embed)

    print("Matematikai csoportosítás...")
    # 2. Csoportok kialakítása távolság alapján (a 0.35-ös értéket később lehet finomítani)
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
    # 3. Csoportok beküldése az LLM-nek egyesével
    for label, items in groups.items():
        formatted_list = ""
        for n in items:
            summary_slice = n['summary'][:200].replace('\n', ' ')
            formatted_list += f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {summary_slice}\n"

        prompt = f"""
        Te egy elit hírszerkesztő vagy. A feladatod a hírek csoportosítása és pontozása.
        
        SZABÁLYOK:
        1. Csak azokat a híreket tartsd meg a csoportban, amelyek TÉNYLEG ugyanarról az eseményről szólnak (Helyszín-elv!).
        2. PONTOZÁSI LOGIKA (Minden mező 1-10):
            - relevance: Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából.
            - impact: Az esemény súlya (pl. háború, nagyvállalati csőd = 10; kisebb nyilatkozat = 3).
            - novelty: Mennyire tartalmaz új, eddig ismeretlen információt.
        3. Kategorizáld a híreket.
            KATEGÓRIÁK:
            - HAZAI: Magyarországi esemény, vagy külföldi esemény ami KÖZVETLENÜL érinti Magyarországot (pl. EU döntés rólunk).
            - GLOBÁLIS: Világszintű nagy hír (USA választás, világgazdaság, háborúk).
            - EGYÉB: Fontos, de távolabbi vagy specifikusabb hírek.
        4. A válasz CSAK egy érvényes JSON objektum legyen!
        
        VÁRT JSON FORMÁTUM:
        {{
            "name": "Esemény neve",
            "category": "HAZAI",
            "scores": {{"relevance": 9, "impact": 8, "novelty": 10}},
            "ids": [ide jönnek az egyező ID-k]
        }}
        
        Hírek:
        {formatted_list}
        """

        # Az új, egyszerűsített hívás: is_json_task=True kényszeríti a JSON-t és a sima Flash modellt
        ai_response = safe_generate_content(prompt, is_json_task=True)
        
        try:
            # Rögtön feldolgozzuk a JSON-t
            clean_json = ai_response.strip().replace('```json', '').replace('```', '').strip()
            cluster_data = json.loads(clean_json)
            if cluster_data and cluster_data.get('ids'):
                final_clusters.append(cluster_data)
        except Exception as e:
            print(f"Hiba a csoport feldolgozásánál: {e}")

    # A régi parse_clusters miatt visszaalakítjuk stringgé a teljes listát
    return json.dumps(final_clusters)
    
def cluster_news_old(news_pool):
    formatted_list = ""
    for n in news_pool:
        summary_slice = n['summary'][:200].replace('\n', ' ')
        formatted_list += f"ID:{n['id']} | CÍM: {n['title']} | KIVONAT: {summary_slice}\n"

    prompt = f"""
    Te egy elit hírszerkesztő vagy, aki csak a legfontosabb gazdasági és politikai hírekre koncentrál. 
    A feladatod a hírek csoportosítása, pontozása és a lényegtelen zaj kiszűrése.
    Használd a CÍMET és a KIVONATOT is az esemény pontos azonosításához és a helyszín meghatározásához.

    SZABÁLYOK:
    1. KONKRÉT ESEMÉNY: Csak azokat a híreket tedd egy csoportba, amelyek TÉNYLEG ugyanarról a konkrét eseményről szólnak.
    2. HELYSZÍN-ELV: Ha két hír helyszíne eltér, NE vond össze őket iparági hasonlóság miatt!
       - TILOS: Kongói bánya + Debreceni akkugyár = KÉT KÜLÖN CSOPORT.
       - SZABAD: Ukrán pénzszállító + Ukrán miniszteri reakció = EGY CSOPORT (közvetlen ok-okozati kapcsolat).
    3. PONTOZÁSI LOGIKA (Minden mező 1-10):
        - relevance: Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából.
        - impact: Az esemény súlya (pl. háború, nagyvállalati csőd = 10; kisebb nyilatkozat = 3).
        - novelty: Mennyire tartalmaz új, eddig ismeretlen információt.
    4. Kategorizáld a híreket.
    KATEGÓRIÁK:
    - HAZAI: Magyarországi esemény, vagy külföldi esemény ami KÖZVETLENÜL érinti Magyarországot (pl. EU döntés rólunk).
    - GLOBÁLIS: Világszintű nagy hír (USA választás, világgazdaság, háborúk).
    - EGYÉB: Fontos, de távolabbi vagy specifikusabb hírek.
    5. A válasz CSAK egy érvényes JSON lista legyen, semmi más szöveg!

    VÁRT JSON FORMÁTUM:
    [
      {{
        "name": "Esemény neve (Helyszín)",
        "category": "HAZAI",
        "scores": {{
            "relevance": 9,
            "impact": 8,
            "novelty": 10
        }},
        "ids": [1, 2]
      }}
    ]

    Hírek listája:
    {formatted_list}
    """

    response = safe_generate_content(prompt, True)
    return response

def parse_clusters(ai_response):
    try:
        clean_json = ai_response.strip().replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        
        filtered = []
        for c in data:
            s = c.get('scores', {})
            # Súlyozott átlag számítása: a relevancia és a hatás fontosabb, mint a novelty
            # Képlet: (Relevancia * 0.4) + (Hatás * 0.4) + (Újdonság * 0.2)
            weighted_score = (s.get('relevance', 0) * 0.4) + \
                             (s.get('impact', 0) * 0.4) + \
                             (s.get('novelty', 0) * 0.2)
            
            # Mentjük a kiszámolt pontot a rendezéshez
            c['total_score'] = round(weighted_score, 1)
            
            # Csak akkor engedjük át, ha a súlyozott pontszám eléri a 6-ot
            if weighted_score >= 6:
                filtered.append(c)
        
        filtered.sort(key=lambda x: x['total_score'], reverse=True)
        return filtered
    except Exception as e:
        print(f"JSON Parse/Scoring Hiba: {e}")
        return []

def summarize_event(cluster_name, ids, news_pool):
    """Második fázis: Egy adott csoport híreiből készít egyetlen összefoglalót."""
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
    """
    Feldarabolja az üzenetet 3900 karakterenként a legközelebbi új sornál,
    és minden részt ellát egy (X/Y) sorszámmal.
    """
    MAX_CHARS = 3900
    
    # Ha belefér egybe, csak a sima fejlécet kapja
    if len(text) <= MAX_CHARS:
        bot.send_message(chat_id, f"🗞 AI HÍRELEMZÉS (1/1)\n\n{text}")
        return

    # Kiszámoljuk a darabokat (közelítőleg)
    # Először listába gyűjtjük a részeket, hogy tudjuk a végleges darabszámot
    parts = []
    temp_text = text
    
    while temp_text:
        if len(temp_text) <= MAX_CHARS:
            parts.append(temp_text.strip())
            break
        
        # Keressük az utolsó dupla sortörést (bekezdés végét) a 3900. karakter előtt
        split_index = temp_text.rfind('\n\n', 0, MAX_CHARS)
        
        # Ha nincs dupla, keressünk sima sortörést
        if split_index == -1:
            split_index = temp_text.rfind('\n', 0, MAX_CHARS)
        
        # Ha így sincs, vágjuk le fixen
        if split_index == -1:
            split_index = MAX_CHARS
            
        parts.append(temp_text[:split_index].strip())
        temp_text = temp_text[split_index:].strip()

    # Most elküldjük a részeket a sorszámozott fejléccel
    total_parts = len(parts)
    for i, part in enumerate(parts, 1):
        header = f"🗞 AI HÍRELEMZÉS ({i}/{total_parts})\n\n"
        bot.send_message(chat_id, header + part)

def generate_rss_file(reports, filename="rss_output.xml"):
    """Létrehoz egy RSS feedet az összefoglalt hírekből."""
    fg = FeedGenerator()
    fg.id('https://github.com/your-repo/ai-news-agent')
    fg.title('AI Hírelemző Összefoglaló')
    fg.author({'name': 'Gemini AI Agent'})
    fg.link(href='https://github.com/green-paw', rel='alternate')
    fg.language('hu')
    fg.description('Napi politikai és gazdasági összefoglalók több forrás alapján')

    for report in reports:
        # A jelentés első sorát (a címet) használjuk az RSS bejegyzés címének
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
    
    # 1. Csoportosítás JSON formátumban
    # A manual_config-ot itt üresen hagyjuk, hogy a JSON kényszerítést használja
    cluster_json_raw = cluster_news(news_pool)
    clusters = parse_clusters(cluster_json_raw)

    if not clusters:
        print("Nem születtek releváns hírcsoportok (vagy JSON hiba történt).")
        return

    # 2. Szétválogatás kategóriák szerint (Dictionary-k listáját kapjuk)
    hazai = [c for c in clusters if c.get('category') == 'HAZAI'][:10]
    globalis = [c for c in clusters if c.get('category') == 'GLOBÁLIS'][:10]
    egyeb = [c for c in clusters if c.get('category') == 'EGYÉB'][:10]

    final_reports = []
    
    # Segédfüggvény a szekciók feldolgozásához és címkézéséhez
    def process_section(section_list, section_title):
        if section_list:
            final_reports.append(f"--- {section_title} ({len(section_list)} esemény) ---")
            for item in section_list:
                # Az item['name'] és item['ids'] a JSON-ból jön
                report = summarize_event(item['name'], item['ids'], news_pool)
                final_reports.append(report)

    # 3. Összefoglalók generálása a kért sorrendben
    print("Összefoglalók készítése szekciónként...")
    process_section(hazai, "MAGYARORSZÁG ÉS RELEVÁNS HÍREK")
    process_section(globalis, "KIEMELT GLOBÁLIS ESEMÉNYEK")
    process_section(egyeb, "EGYÉB FONTOS HÍREK A VILÁGBÓL")

    # 4. Kimenetek kezelése
    reports_count = len(clusters) # Az eredeti csoportok száma (a címek nélkül)
    
    if len(final_reports) > 0:
        print(f"Kész! {reports_count} releváns esemény összefoglalva.")
        
        # Teljes üzenet összefűzése a Telegramhoz
        full_message = "\n\n".join(final_reports)
        
        # Küldés Telegramra darabolva
        try:
            print("Küldés Telegramra...")
            send_split_message(config.TELEGRAM_CHAT_ID, full_message)
        except Exception as e:
            print(f"Telegram hiba: {e}")

        # RSS fájl mentése
        try:
            generate_rss_file(final_reports, "rss_output.xml")
        except Exception as e:
            print(f"RSS hiba: {e}")
    else:
        print("A szűrési feltételeknek (Score >= 6) egyetlen hír sem felelt meg.")
        
if __name__ == "__main__":
    main()
