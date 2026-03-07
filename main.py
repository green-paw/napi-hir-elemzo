import os
import feedparser
import re
import time
import json
import requests
from google import genai
from datetime import datetime

# --- BEÁLLÍTÁSOK ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"

client = genai.Client(api_key=GOOGLE_API_KEY)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_history(history, new_entries):
    # Az utolsó 15 elemzést tároljuk el
    combined = history + new_entries
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(combined[-15:], f, ensure_ascii=False, indent=2)

def send_telegram_chunked(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    MAX_LENGTH = 3900
    clean_text = text.replace('*', '').replace('_', '').replace('`', '')
    chunks = [clean_text[i:i+MAX_LENGTH] for i in range(0, len(clean_text), MAX_LENGTH)]

    for i, chunk in enumerate(chunks):
        header = f"<b>🗞 Napi Mélyelemzés ({i+1}/{len(chunks)})</b>\n\n"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": header + chunk, "parse_mode": "HTML"}
        requests.post(url, data=payload)
        time.sleep(1)

def analyze_single_news(news_item, past_context):
    prompt = f"""
    Te egy szigorú, tényalapú magyar médiaelemző vagy. 
    HÍR: {news_item['title']}
    FORRÁSOK: {news_item['summary']}
    
    ELŐZMÉNYEK (Múltbeli hírek):
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

    STÍLUS:
    - Ne használj Markdownt! 
    - Kerüld a bullshitet és a felesleges körmondatokat. 
    - Az elemzés legyen lényegre törő, de alapos (maximum 10 mondat hírenként).
    """
    try:
        response = client.models.generate_content(model='gemini-flash-lite-latest', contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Hiba: {e}"

def main():
    history = load_history()
    # Az utolsó 5 elemzést adjuk át kontextusnak
    past_context = "\n".join([f"- {h['date']}: {h['title']}" for h in history[-5:]])
    
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    blacklist = ["foci", "bajnokság", "celeb", "horoszkóp", "bulvár", "recept"]
    
    scored_news = []
    for entry in feed.entries:
        if any(word in entry.title.lower() for word in blacklist): continue
        score = len(re.findall(r'<li>', entry.summary))
        scored_news.append({"title": entry.title, "summary": entry.summary, "score": score})

    top_news = sorted(scored_news, key=lambda x: x['score'], reverse=True)[:6]
    
    new_history_entries = []
    full_message = ""
    
    for news in top_news:
        analysis = analyze_single_news(news, past_context)
        if "SKIP" not in analysis:
            full_message += f"📌 {news['title'].upper()}\n{analysis}\n\n"
            new_history_entries.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "title": news['title'],
                "summary": analysis[:200] # Csak rövid kivonat a memóriához
            })
        time.sleep(3)

    if full_message:
        send_telegram_chunked(full_message)
        save_history(history, new_history_entries)

if __name__ == "__main__":
    main()
