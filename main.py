import os
import feedparser
import google.genai as genai # A Groq helyett ezt használjuk
import requests

# Beállítások
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Gemini konfigurálása
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemma-3n')
#model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-09-2025')

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

def analyze_today():
    # Hírek lekérése
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    top_titles = [entry.title for entry in feed.entries[:5]]
    
    # Itt a Gemini-nek szóló magyar nyelvű instrukció
    prompt = f"""
    Te egy tapasztalt magyar médiaelemző vagy. 
    Itt van a mai 5 vezető hír címe Magyarországról:
    {top_titles}

    Készíts egy profi, rövid elemzést a Telegramra. 
    Minden hírnél fejtsd ki:
    1. Mi a valódi esemény?
    2. Hogyan tálalja ezt a kormányközeli média (pl. sikerpropaganda vagy ellenségkép)?
    3. Hogyan tálalja a független média (pl. kritikai észrevételek vagy elhallgatott részletek)?
    
    Használj magyaros kifejezéseket! Legyen tömör.
    Kérlek, ne használj bonyolult Markdown formázást, csak a vastagítást (csillagokkal) és egyszerű listákat.
    """

    # Gemini hívása
    response = model.generate_content(prompt)
    valasz = response.text
    
    send_telegram(f"🇭🇺 *Napi Magyar Hírelemző (Gemini)*\n\n{valasz}")

if __name__ == "__main__":
    analyze_today()
