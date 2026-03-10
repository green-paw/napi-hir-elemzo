import config
import feedparser
import re
import telebot # Feltételezve, hogy a pyTelegramBotAPI-t használod
from google import genai
import time
from google.genai import errors

# --- Konfiguráció inicializálása ---
client = genai.Client(api_key=config.GOOGLE_API_KEY)
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def safe_generate_content(prompt):
    """Újrapróbálkozó függvény 503-as hiba esetén."""
    for attempt in range(3): # Max 3 próbálkozás
        try:
            response = client.models.generate_content(
                model=config.MODEL_ID,
                contents=prompt,
                config={'temperature': 0.1}
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
    formatted_list = "\n".join([f"ID:{i['id']} | {i['title']}" for i in news_pool])

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

    Hírek listája:
    {formatted_list}

    VÁLASZ FORMÁTUMA (Kizárólag):
    SCORE: [pontszám] | ESEMÉNY NEVE (HELYSZÍN): [ID1, ID2]
    """

    response = safe_generate_content(prompt)
    return response

def parse_clusters(ai_response):
    clusters = []
    # Soronként nézzük át az AI válaszát
    lines = ai_response.strip().split('\n')
    
    for line in lines:
        try:
            # Reguláris kifejezéssel kikeressük a pontszámot, a nevet és az ID-kat
            # Minta: SCORE: 9 | ESEMÉNY (HELYSZÍN): [1, 2]
            match = re.search(r'SCORE:\s*(\d+)\s*\|\s*(.*?):\s*\[(.*?)\]', line)
            
            if match:
                score = int(match.group(1))
                name = match.group(2).strip()
                # Az ID-kat listává alakítjuk
                ids = [int(i) for i in re.findall(r'\d+', match.group(3))]
                
                # --- EZ A KRITIKUS SZŰRÉS ---
                if score >= 6:
                    clusters.append({
                        'score': score, 
                        'name': name, 
                        'ids': ids
                    })
                else:
                    print(f"Kiszűrve alacsony pontszám miatt ({score}): {name}")
                    
        except Exception as e:
            print(f"Hiba a sor feldolgozásakor: {line} | Hiba: {e}")
            continue
            
    # Rendezés: a legfontosabb (legmagasabb pontszám) legyen legelöl
    clusters.sort(key=lambda x: x['score'], reverse=True)
    return clusters

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

    # 4. Küldés Telegramra
    full_message = "\n\n".join(final_reports)
    
    if full_message:
        print("Üzenet küldése Telegramra (darabolva ha szükséges)...")
        try:
            send_split_message(config.TELEGRAM_CHAT_ID, full_message)
            print("Sikeres küldés!")
        except Exception as e:
            print(f"Telegram hiba: {e}")
    else:
        print("Nem született releváns összefoglaló.")

if __name__ == "__main__":
    main()
