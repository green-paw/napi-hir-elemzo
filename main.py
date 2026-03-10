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
    Te egy precíz hírszerkesztő vagy. A feladatod a hírek csoportosítása szigorú eseményalapú logika szerint.

    SZABÁLYOK:
    1. KONKRÉT ESEMÉNY: Csak azokat a híreket tedd egy csoportba, amelyek TÉNYLEG ugyanarról a konkrét eseményről szólnak.
    2. HELYSZÍN ALAPÚ SZÉTVÁLASZTÁS: Ha két hír helyszíne jelentősen eltér (pl. más ország), NE vond össze őket, még ha az iparág azonos is (pl. akkumulátor ipar). Kivétel ha több ország közötti kommunikáció vagy konfliktus okán függenek össze az események.
        Példák:
            - akkumulátor gyár Debrecenben és Kongói bányabaleset ahol akkuhoz bányásznak anyagot két külön csoport.
            - Magyarországon feltartóztatott Ukrán pénzszállító és az Ukrán miniszterelnök nyilatkozata az akcióról ugyanaz az csoport.
    3. RANGSOROLÁS (SCORE): Minden csoporthoz rendelj egy 1-10 közötti pontszámot:
        - 10: Rendkívüli esemény (háború, kormányváltás, gazdasági összeomlás).
        - 7-9: Fontos politikai/gazdasági hír (kamatdöntés, elnöki nyilatkozat, nagyvállalati botrány).
        - 4-6: Átlagos napi hír (útlezárás, kisebb törvénymódosítás).
        - 1-3: Érdekesség, technikai jellegű hír.
       Fontos: Egy egyforrásos hír is kaphat 10-est, ha a tartalma súlyos!

    A válaszból szűrd ki az 5-ös fontossági pont alatti híreket, és a bulvárt.
   
    Hírek listája:
    {formatted_list}

    A válaszod formátuma szigorúan: 
    ESEMÉNY NEVE (HELYSZÍN): [ID1, ID2]
    
    Példa: 
    KAMATDÖNTÉS (BUDAPEST): [4, 8, 12]
    BÁNYABALESET (KONGÓ): [15, 22]
    """

    response = safe_generate_content(prompt)
    return response

def parse_clusters(ai_response):
    """Kinyeri az ID-kat az AI válaszából egy szótárba."""
    clusters = {}
    lines = ai_response.strip().split('\n')
    for line in lines:
        if ':' in line and '[' in line:
            parts = line.split(':')
            name = parts[0].strip()
            ids = re.findall(r'\d+', parts[1])
            if ids:
                clusters[name] = [int(i) for i in ids]
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

    # 3. Összefoglalás (Reduce)
    final_reports = []
    print(f"{len(clusters)} esemény összefoglalása folyamatban...")
    for name, ids in clusters.items():
        try:
            report = summarize_event(name, ids, news_pool)
            final_reports.append(f"📌 {report}")
        except Exception as e:
            print(f"Hiba az összefoglalásnál ({name}): {e}")

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
