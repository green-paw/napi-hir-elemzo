import os
import feedparser
from genai import Client # Ez az új SDK
import requests

# Beállítások
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Az új kliens inicializálása
client = Client(api_key=GOOGLE_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": text, 
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

def analyze_today():
    # Hírek lekérése
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    top_titles = [entry.title for entry in feed.entries[:5]]
    
    prompt = f"""
    Magyar médiaelemző vagy. Elemezd ezt az 5 hírt kétoldalú nézőpontból: {top_titles}. 
    Mondd meg mi a tény, és mi a várható keretezés a két oldalon.
    Használj emoji-kat és Markdown formázást.
    """

    # ÚJ SZINTAXIS: A kliens hívja a generálást
    response = client.models.generate_content(
        model='gemini-2.0-flash', # Az új SDK-ban már nem kell a 'models/' előtag
        contents=prompt
    )
    
    valasz = response.text
    send_telegram(f"🗞 *Napi Hírelemzés (Új GenAI SDK)*\n\n{valasz}")

if __name__ == "__main__":
    analyze_today()
