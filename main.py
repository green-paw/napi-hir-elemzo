import config
import feedparser
import re
import telebot # Feltételezve, hogy a pyTelegramBotAPI-t használod
from google import genai
import time
from google.genai import errors
from feedgen.feed import FeedGenerator
from datetime import datetime
import pytz
import json

# --- Konfiguráció inicializálása ---
client = genai.Client(api_key=config.GOOGLE_API_KEY)
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def safe_generate_content(prompt, manual_config):
    """Újrapróbálkozó függvény 503-as hiba esetén."""
    config = manual_config if manual_config else {'temperature': 0.1}
    for attempt in range(3): # Max 3 próbálkozás
        try:
            response = client.models.generate_content(
                model=config.MODEL_ID,
                contents=prompt,
                config=config
            )
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

def cluster_news(news_pool):
    """Első fázis: Az LLM csak az ID-kat és címeket látja, és csoportokat alkot."""
    formatted_list = "\n".join([f"ID:{i['id']} | {i['title']} | {i['summary'][:100]}" for i in news_pool])

    prompt = f"""
    Te egy elit hírszerkesztő vagy, aki csak a legfontosabb gazdasági és politikai hírekre koncentrál. 
    A feladatod a hírek csoportosítása, pontozása és a lényegtelen zaj kiszűrése.

    SZABÁLYOK:
    1. KONKRÉT ESEMÉNY: Csak azokat a híreket tedd egy csoportba, amelyek TÉNYLEG ugyanarról a konkrét eseményről szólnak.
    2. HELYSZÍN-ELV: Ha két hír helyszíne eltér, NE vond össze őket iparági hasonlóság miatt!
       - TILOS: Kongói bánya + Debreceni akkugyár = KÉT KÜLÖN CSOPORT.
       - SZABAD: Ukrán pénzszállító + Ukrán miniszteri reakció = EGY CSOPORT (közvetlen ok-okozati kapcsolat).
    3. RANGSOROLÁS ÉS SZŰRÉS:
       - 10: Rendkívüli (háború, kormányváltás, gazdasági krach).
       - 7-9: Kiemelt hír (kamatdöntés, elnöki nyilatkozat).
       - 6: Fontos hír (miniszteri nyilatkozat, jelentős törvénymódosítás).
       - 1-5: egyéb hírek (bulvár, balesetek, kis színes hírek, sporthírek)
    FIGYELEM: Minden hírt, ami 6 pont alatti (bulvár, balesetek, kis színes hírek, sporthírek), szigorúan dobj el! Ne listázd ki őket!
    4. Kategorizáld a híreket.
    KATEGÓRIÁK:
    - HAZAI: Magyarországi esemény, vagy külföldi esemény ami KÖZVETLENÜL érinti Magyarországot (pl. EU döntés rólunk).
    - GLOBÁLIS: Világszintű nagy hír (USA választás, világgazdaság, háborúk).
    - EGYÉB: Fontos, de távolabbi vagy specifikusabb hírek.
    5. A válasz CSAK egy érvényes JSON lista legyen, semmi más szöveg!

    VÁLASZ FORMÁTUMA (Szigorúan):
    [
      {{"score": 9, "category": "HAZAI", "name": "Kamatdöntés (Budapest)", "ids": [1, 5]}},
      {{"score": 7, "category": "GLOBÁLIS", "name": "Elnökválasztás (Washington)", "ids": [3]}}
    ]

    Hírek listája:
    {formatted_list}
    """

    response = safe_generate_content(prompt, {
        'temperature': 0.0,
        'response_mime_type': 'application/json' # Ez kényszeríti a JSON-t!
    })
    return response

def parse_clusters(ai_response):
    try:
        clean_json = ai_response.strip().replace('```json', '').replace('```', '').strip()
        clusters = json.loads(clean_json)
        
        filtered_clusters = [c for c in clusters if c.get('score', 0) >= 6]
        filtered_clusters.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return filtered_clusters
    except Exception as e:
        print(f"JSON feldolgozási hiba: {e}\nAz eredeti válasz: {ai_response}")
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
    fg.link(href='https://github.com/your-repo', rel='alternate')
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
        fe.pubDate(datetime.now(pytz.utc))

    fg.rss_file(filename)
    print(f"RSS feed sikeresen elmentve: {filename}")

def main():
    # 1. Adatgyűjtés
    news_pool = fetch_news()
    if not news_pool:
        print("Nem sikerült híreket beolvasni.")
        return

    # 2. Csoportosítás (Map)
    print(f"{len(news_pool)} hír elemzése és csoportosítása...")
    cluster_text = cluster_news(news_pool)
    print(f"Csoportosítás eredménye:\n{cluster_text}")
    clusters = parse_clusters(cluster_text)

    #3. summarize
    final_reports = []
    for item in clusters:
        report = summarize_event(item['name'], item['ids'], news_pool)
        final_reports.append(report)

    if len(final_reports) > 0:
        # 4. Küldés Telegramra
        full_message = "\n\n".join(final_reports)
    
        if full_message:
            print("Üzenet küldése Telegramra (darabolva ha szükséges)...")
            try:
                send_split_message(config.TELEGRAM_CHAT_ID, full_message)
                print("Sikeres küldés!")
            except Exception as e:
                print(f"Telegram hiba: {e}")

        # 5. RSS output
        try:
            generate_rss_file(final_reports, "rss_output.xml")
        except Exception as e:
            print(f"Hiba az RSS fájl írásakor: {e}")
    else:
        print("Nem született releváns összefoglaló.")

if __name__ == "__main__":
    main()
