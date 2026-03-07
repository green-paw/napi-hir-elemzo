import os
import feedparser
import re
import time
import requests
from google import genai

# --- BEÁLLÍTÁSOK ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = genai.Client(api_key=GOOGLE_API_KEY)

def send_telegram_chunked(text):
    """Szétvágja az üzenetet és elküldi több részletben, ha szükséges."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    MAX_LENGTH = 3900 # Biztonsági ráhagyással
    
    # Tisztítás a formázási hibák elkerülésére
    clean_text = text.replace('*', '').replace('_', '').replace('`', '')

    chunks = []
    while len(clean_text) > 0:
        if len(clean_text) <= MAX_LENGTH:
            chunks.append(clean_text)
            break
        split_at = clean_text.rfind('\n', 0, MAX_LENGTH)
        if split_at == -1: split_at = MAX_LENGTH
        chunks.append(clean_text[:split_at])
        clean_text = clean_text[split_at:].lstrip()

    for i, chunk in enumerate(chunks):
        header = f"<b>🗞 Napi Hírelemzés ({i+1}/{len(chunks)})</b>\n\n"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": header + chunk,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, data=payload)
        if resp.status_code != 200: # Fallback ha a HTML elszállna
            del payload["parse_mode"]
            requests.post(url, data=payload)
        time.sleep(1) # Kis szünet az üzenetek között

def analyze_single_news(news_item):
    prompt = f"""
    Te egy magyar médiaelemző vagy. Elemezd ezt az egy hírt mélyen:
    CÍM: {news_item['title']}
    FORRÁSOK: {news_item['summary']}

    FELADAT ÉS SZABÁLYOK:
    1. SZŰRÉS: Ha a hír bulvár, technikai jellegű, sport, vagy tisztán kattintásvadász (clickbait), válaszolj ennyit: "SKIP".
    2. MANIPULÁCIÓ SZŰRÉSE: Távolíts el minden érzelmi töltetű jelzőt és manipulatív fordulatot. Csak a száraz tényekre és összefüggésekre koncentrálj.
    
    STRUKTÚRA (Elemezd részletesen):
    - KONZERVATÍV NARRATÍVA: Hogyan keretezik a kormányközeli lapok? Mi a stratégiai üzenetük?
    - KRITIKUS NARRATÍVA: Mit emel ki a kritikus sajtó? Milyen hiányosságra mutatnak rá?
    - GAZDASÁGI HATÁS: Milyen pénzügyi, piaci vagy megélhetési következménye van ennek?
    - NEMZETKÖZI KONTEXTUS: Hogyan illeszkedik ez a globális folyamatokba (EU, NATO, szomszédok)?
    - TÉNY: Mi a megtisztított, objektív valóság?

    SZIGORÚ SZABÁLYOK: 
    - Ne használj Markdownt! 
    - Kerüld a bullshitet és a felesleges körmondatokat. 
    - Az elemzés legyen lényegre törő, de alapos (kb. 10-15 mondat hírenként).
    """
    try:
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        valasz = response.text.strip()
        
        if "SKIP" in valasz or len(valasz) < 50:
            return ""
            
        return f"📌 {news_item['title'].upper()}\n{valasz}\n\n"
    except Exception as e:
        return f"❌ Hiba: {e}\n\n"

def main():
    print("Hírek begyűjtése...")
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    
    # Tiltólista a nem kívánt kategóriákhoz
    blacklist = ["foci", "bajnokság", "mérkőzés", "celeb", "vlog", "okostelefon", "teszt", "recept", "horoszkóp", "bulvár"]
    
    scored_news = []
    for entry in feed.entries:
        # Cím ellenőrzése a tiltólistával
        if any(word in entry.title.lower() for word in blacklist):
            continue
            
        score = len(re.findall(r'<li>', entry.summary))
        scored_news.append({"title": entry.title, "summary": entry.summary, "score": score})

    # A 7 legfontosabb hír (Map-Reduce folyamat)
    top_news = sorted(scored_news, key=lambda x: x['score'], reverse=True)[:7]
    
    full_analysis = ""
    for i, news in enumerate(top_news):
        print(f"Elemzés: {i+1}/{len(top_news)}")
        analysis = analyze_single_news(news)
        full_analysis += analysis
        time.sleep(3) # Kvóta-védelem (RPM limit miatt)

    if full_analysis:
        send_telegram_chunked(full_analysis)
        print("Kész! Üzenetek elküldve.")

if __name__ == "__main__":
    main()
