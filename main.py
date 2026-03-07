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
    """Egyetlen hír mélyelemzése."""
    prompt = f"""
    Te egy magyar médiaelemző vagy. Elemezd ezt az egy hírt mélyen:
    CÍM: {news_item['title']}
    FORRÁSOK: {news_item['summary']}

    Mutasd be a kormányközeli és a kritikus narratívát:
    1. KONZERVATÍV NARRATÍVA: Mi a keretezés?
    2. KRITIKUS NARRATÍVA: Mit emelnek ki?
    3. TÉNY: Mi a közös alap?

    SZIGORÚ SZABÁLY: Csak a megadott adatokból dolgozz. Ne használj Markdownt!
    Válaszolj tömören, 4-5 mondatban összesen.
    """
    try:
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        return f"📌 {news_item['title'].upper()}\n{response.text}\n\n"
    except Exception as e:
        return f"❌ Hiba az elemzés során ({news_item['title'][:20]}): {e}\n\n"

def main():
    print("Hírek begyűjtése...")
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    
    scored_news = []
    for entry in feed.entries:
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
