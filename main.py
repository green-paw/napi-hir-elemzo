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
                clean_summary = re.sub('<[^<]+?>', '', summary)[:400]
                
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
    Te egy precíz hírszerkesztő algoritmus vagy. 
    Kizárólag azokat a híreket csoportosítsd, amelyek ugyanarról a konkrét eseményről szólnak!

    Szabályok:
    1. Ha két hír csak témájában hasonló (pl. mindkettő külpolitika), de különböző események, NE tedd őket egy csoportba!
    2. Egy csoportba csak az kerülhet, ami ugyanazt a történést dolgozza fel különböző forrásokból.
    3. Ami egyedi esemény és nincs párja, azt hagyd ki a csoportosításból (vagy tedd egyedül egy csoportba).
    4. Szigorúan tilos "Vegyes" vagy "Külpolitikai összefoglaló" típusú gyűjtőcsoportokat létrehozni.
    5. Csak a releváns gazdasági és politikai eseményeket tartsd meg!

    Hírek listája:
    {formatted_list}

    A válaszod formátuma szigorúan és kizárólag ennyi legyen (minden esemény új sor):
    Esemény rövid neve: [ID1, ID2, ID3]
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
    
    Írj belőlük egyetlen, tárgyilagos, rövid magyar nyelvű összefoglalót (max 3 mondat).
    A végén tüntesd fel a forrásokat így: (Forrás: {sources_str})
    Szigorúan tilos a Markdown formázás (vastagítás, csillagok, dőlt betű)!
    """

    response = safe_generate_content(prompt)
    return response

def main():
    # 1. Adatgyűjtés
    news_pool = fetch_news()
    if not news_pool:
        print("Nem sikerült híreket beolvasni.")
        return

    # 2. Csoportosítás (Map)
    print(f"{len(news_pool)} hír elemzése és csoportosítása...")
    cluster_text = cluster_news(news_pool)
    print(f"Csoportosítás eredménye:\n{response}")
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
        print("Üzenet küldése Telegramra...")
        try:
            bot.send_message(config.TELEGRAM_CHAT_ID, full_message)
            print("Sikeres küldés!")
        except Exception as e:
            print(f"Telegram hiba: {e}")
            # Biztonsági mentés konzolra, ha a Telegram elszállna
            print(full_message)
    else:
        print("Nem született releváns összefoglaló a mai hírekből.")

if __name__ == "__main__":
    main()
