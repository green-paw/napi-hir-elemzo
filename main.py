import os
import feedparser
from groq import Groq
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Groq(api_key=GROQ_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": int(TELEGRAM_CHAT_ID), "text": text, "parse_mode": "Markdown"}
    response = requests.post(url, data=payload)
    print(f"Telegram válasz: {response.status_code} - {response.text}") # Ez kiírja a hibát!
    return response
    
def analyze_today():
    # 1. Lekérjük a Google News magyarországi vezető híreit
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    top_titles = [entry.title for entry in feed.entries[:5]]
    
    # 2. Megkérjük az AI-t az elemzésre
    prompt = f"""
    Te egy pártatlan média-elemző ágens vagy. 
    Itt van a mai 5 legfontosabb hír címe Magyarországról:
    {top_titles}

    Készíts egy rövid, átlátható elemzést a Telegramra. 
    Minden hírnél térj ki arra:
    - Mi a puszta tény?
    - Hogyan tálalhatja ezt a 'kormányközeli' vs. 'független' média? (keretezési különbségek)
    - Csatold a linkeket minden hír alatt.
    
    Használj Markdown formázást! Ne használj emojikat!
    """

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )
    
    valasz = completion.choices[0].message.content
    
    # 3. Küldés a telefonra
    send_telegram(f"*Napi Hírelemző Jelenti*\n\n{valasz}")
    print("Siker! Nézd meg a Telegramodat.")

# Indítás
analyze_today()
