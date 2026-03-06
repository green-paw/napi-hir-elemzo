import feedparser
from google import genai
import requests
import re
import os

# Beállítások
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Az új kliens inicializálása
client = genai.Client(api_key=GOOGLE_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # 1. BIZTONSÁGI VÁGÁS: A Telegram limitje 4096, mi megállunk 3900-nál
    if len(text) > 3900:
        text = text[:3900] + "\n\n[...VÁGVA A HOSSZ MIATT...]"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown" # Ha továbbra is 400-as hiba van, töröld ki ezt a sort!
    }
    
    response = requests.post(url, data=payload)
    print(f"Telegram status: {response.status_code}, Response: {response.text}")
    
    # Ha a Markdown elrontja, küldjük el sima szövegként (fallback)
    if response.status_code != 200:
        del payload["parse_mode"]
        requests.post(url, data=payload)

def analyze_trending_news():
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    
    scored_news = []
    for entry in feed.entries:
        # Relevancia pontszám a források száma alapján
        source_count = len(re.findall(r'<li>', entry.summary))
        scored_news.append({
            "title": entry.title,
            "summary": entry.summary,
            "score": source_count
        })

    # Top 8 hír (kicsit kevesebb, hogy biztosan beférjen a 4000 karakterbe)
    top_news = sorted(scored_news, key=lambda x: x['score'], reverse=True)[:8]

    context = ""
    for i, news in enumerate(top_news, 1):
        context += f"{i}. HÍR: {news['title']}\nFORRÁSOK: {news['summary']}\n\n"

    prompt = f"""
    Te egy magyar médiaelemző szoftver vagy. Az alábbi 10 hír ma a legmeghatározóbb a magyar sajtóban (relevancia szerint rangsorolva):
    {context}

    FELADAT (Narratíva-rekonstrukció):
    Válaszd ki a listából azokat, amelyeknek komoly politikai vagy gazdasági súlya van (hagyd ki a bulvárt, ha van benne).
    Mutasd be a két pólust:

    📌 [Hír címe]
    - Jobb oldal: Hogyan keretezik? Kulcsszavak?
    - Bal oldal: Mit emelnek ki/mit kritizálnak?
    - KÖZÖS METSZET: Mi a puszta tény?

    SZIGORÚ SZABÁLYOK:
    1. Csak a megadott forrásokból dolgozz! Ha nincs adat, írd: "Nincs adat".
    2. Ne találj ki háttérsztorit.
    3. Tömör, egyszerű markdown.
    4. Maximum 3900 karakter lehet az egész hírösszefoglaló.
    """

    try:
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        valasz = response.text
        send_telegram(f"🗞 *Napi Top Hírelemzés*\n\n{valasz}")
    except Exception as e:
        print(f"Hiba az AI folyamatban: {e}")

def analyze_trending_news_old():
    print("Vezető hírek begyűjtése és rangsorolása...")
    # A főoldali RSS-t használjuk, mert itt csoportosít a Google
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    
    scored_news = []
    
    for entry in feed.entries:
        # Kiszámoljuk a 'fontosságot': hány darab forrást (újságot) említ a leírás?
        # A summary-ben a források listája így néz ki általában: "Forrás 1, Forrás 2, Forrás 3..."
        source_count = len(re.findall(r'<li>', entry.summary)) # A Google <li> elemekbe teszi a forrásokat
        
        scored_news.append({
            "title": entry.title,
            "summary": entry.summary,
            "score": source_count
        })

    # Rangsorolás: a legtöbb forrással rendelkező hírek kerülnek előre
    top_news = sorted(scored_news, key=lambda x: x['score'], reverse=True)[:10]

    # Összefűzzük az adatokat az AI-nak
    context = ""
    for i, news in enumerate(top_news, 1):
        context += f"{i}. HÍR: {news['title']}\nRELEVANCIA (források száma): {news['score']}\nFORRÁSLISTA: {news['summary']}\n\n"

    prompt = f"""
    Te egy magyar médiaelemző szoftver vagy. Az alábbi 10 hír ma a legmeghatározóbb a magyar sajtóban (relevancia szerint rangsorolva):
    {context}

    FELADAT (Narratíva-rekonstrukció):
    Válaszd ki a listából azokat, amelyeknek komoly politikai vagy gazdasági súlya van (hagyd ki a bulvárt, ha van benne).
    Mutasd be a két pólust:

    📌 [Hír címe]
    - Jobb oldal: Hogyan keretezik? Kulcsszavak?
    - Bal oldal: Mit emelnek ki/mit kritizálnak?
    - KÖZÖS METSZET: Mi a puszta tény?

    SZIGORÚ SZABÁLYOK:
    1. Csak a megadott forrásokból dolgozz! Ha nincs adat, írd: "Nincs adat".
    2. Ne találj ki háttérsztorit.
    3. Tömör, egyszerű markdown.
    4. Maximum 3800 karakter lehet az egész hírösszefoglaló.
    """

    print(f"Elemzés indítása a Top 10 legfontosabb hír alapján (Gemini Flash Lite)...")
    
    try:
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
    except Exception as e:
        print(f"Hiba: {e}")

    print(f"AI generált szöveg hossza: {len(response.text)}")
    return response.text

def analyze_today():
    # Hírek lekérése
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    top_titles = [entry.title for entry in feed.entries[:5]]
    
    prompt = f"""
    Magyar médiaelemző vagy. Elemezd ezt az 5 hírt: {top_titles}
    Minden hírnél csak 2-2 mondatod van (tény + keretezés). 
    FIGYELEM: Az egész válaszod ne legyen több 2000 karakternél!
    Használj egyszerű Markdown-t.
    """
    
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=prompt
    )
    
    valasz = response.text
    send_telegram(f"🗞 *Napi Hírelemzés (Új GenAI SDK)*\n\n{valasz}")

if __name__ == "__main__":
    analyze_trending_news()
    
