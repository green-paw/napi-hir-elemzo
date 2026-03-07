import os
import feedparser
import re
import time
import json
import requests
from google import genai
from datetime import datetime

# --- KONFIGURÁCIÓ ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_PAID_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"

# A 2.0 Flash modell használata (gyors és okos)
MODEL_ID = "gemini-2.0-flash" 

client = genai.Client(api_key=GOOGLE_API_KEY)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_history(history, new_entries):
    # Megnövelt memória: az utolsó 30 elemzést tároljuk (ebből válogatunk később)
    combined = history + new_entries
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(combined[-30:], f, ensure_ascii=False, indent=2)

def send_telegram_chunked(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    MAX_LENGTH = 3900
    clean_text = text.replace('*', '').replace('_', '').replace('`', '')
    chunks = [clean_text[i:i+MAX_LENGTH] for i in range(0, len(clean_text), MAX_LENGTH)]

    for i, chunk in enumerate(chunks):
        header = f"<b>🗞 Napi Mélyelemzés ({i+1}/{len(chunks)})</b>\n\n"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": header + chunk, "parse_mode": "HTML"}
        requests.post(url, data=payload)
        time.sleep(0.5) # Gyorsabb küldés

def analyze_single_news(news_item, past_context):
    prompt = f"""
    Te egy vezető magyar média-stratégiai elemző vagy. 
    HÍR: {news_item['title']}
    FORRÁSOK: {news_item['summary']}
    
    KONTEXTUS (Az elmúlt napok eseményei):
    {past_context}

    FELADAT ÉS SZIGORÚ SZABÁLYOK:
    1. TILOS A FELTÉTELEZÉS: Ne használd a "valószínűleg", "vélhetően", "mondhatnák" fordulatokat. 
    2. HA NINCS ADAT: Ha a megadott forrásban egy oldal nem szólal meg, írd: "Nincs fellelhető releváns narratíva." Ne találd ki, mit mondanának!
    3. ÖSSZEHASONLÍTÁS: Ha a hír kapcsolódik az előzményekhez, mutass rá a változásra vagy ellentmondásra. Ha nem kapcsolódik, hagyd ki ezt a részt.
    4. STRUKTÚRA:
       - KONZERVATÍV NARRATÍVA: (Csak ha van adat)
       - KRITIKUS NARRATÍVA: (Csak ha van adat)
       - GAZDASÁGI/NEMZETKÖZI HATÁS: (Száraz tények)
       - TÉNY: (Manipulációmentes mag)
    5. SZŰRÉS: Ha a hír bulvár, technikai jellegű, sport, vagy tisztán kattintásvadász (clickbait), válaszolj ennyit: "SKIP".
    6. MANIPULÁCIÓ SZŰRÉSE: Távolíts el minden érzelmi töltetű jelzőt és manipulatív fordulatot. Csak a száraz tényekre és összefüggésekre koncentrálj.

    STÍLUS: Max 15 mondat, strukturált, professzionális szöveg. Ne használj Markdownt!
    """
    try:
        # Nincs szükség hosszú várakozásra a fizetős tier miatt
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Hiba az elemzésnél: {e}")
        return "SKIP"

def main():
    history = load_history()
    # Megnövelt kontextus: Az utolsó 10 hír címét és lényegét adjuk át
    past_context = "\n".join([f"- {h['date']}: {h['title']}" for h in history[-10:]])
    
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    blacklist = ["foci", "bajnokság", "celeb", "horoszkóp", "bulvár", "recept", "okostelefon"]
    
    scored_news = []
    for entry in feed.entries:
        if any(word in entry.title.lower() for word in blacklist): continue
        score = len(re.findall(r'<li>', entry.summary))
        scored_news.append({"title": entry.title, "summary": entry.summary, "score": score})

    # Most már bátran elemezhetünk 8-10 hírt is
    top_news = sorted(scored_news, key=lambda x: x['score'], reverse=True)[:8]
    
    new_history_entries = []
    full_message = ""
    
    for i, news in enumerate(top_news):
        print(f"Elemzés ({i+1}/{len(top_news)}): {news['title']}")
        analysis = analyze_single_news(news, past_context)
        
        if analysis and "SKIP" not in analysis:
            full_message += f"📌 {news['title'].upper()}\n{analysis}\n\n"
            new_history_entries.append({
                "date": datetime.now().strftime("%m.%d %H:%M"),
                "title": news['title'],
                "summary": analysis[:300] # Hosszabb kivonat a memóriához
            })
        
        # Csak minimális szünet a biztonság kedvéért
        time.sleep(1.5)

    if full_message:
        send_telegram_chunked(full_message)
        save_history(history, new_history_entries)
        print("Sikeres futás és mentés.")

if __name__ == "__main__":
    main()
