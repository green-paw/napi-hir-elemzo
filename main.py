import os
import feedparser
import re
import time
import json
import requests
from google import genai
from datetime import datetime
from feedgen.feed import FeedGenerator # Ne felejtsd el hozzáadni a YAML-hez!

# --- KONFIGURÁCIÓ ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"
MODEL_ID = "gemini-2.5-flash" 

client = genai.Client(api_key=GOOGLE_API_KEY)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_history(history, new_entries):
    combined = history + new_entries
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(combined[-40:], f, ensure_ascii=False, indent=2)

def generate_rss(entries):
    """Saját RSS feed generálása a GitHub Pages-hez."""
    fg = FeedGenerator()
    fg.id('ai-strat-news-hu')
    fg.title('AI Stratégiai Hírelemzés')
    fg.author({'name': 'Gemini AI'})
    fg.link(href='https://news.google.com', rel='alternate')
    fg.description('Rövidített, többoldalú napi hírelemzések')
    
    for entry in entries:
        fe = fg.add_entry()
        fe.id(entry['title'])
        fe.title(entry['title'])
        fe.description(entry['summary'])
        fe.pubDate(datetime.now().astimezone())
    
    fg.rss_file('rss_output.xml')

def send_telegram_chunked(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    MAX_LENGTH = 3900
    clean_text = text.replace('*', '').replace('_', '').replace('`', '')
    chunks = []
    while len(clean_text) > 0:
        if len(clean_text) <= MAX_LENGTH:
            chunks.append(clean_text); break
        split_at = clean_text.rfind('\n', 0, MAX_LENGTH)
        if split_at == -1: split_at = MAX_LENGTH
        chunks.append(clean_text[:split_at])
        clean_text = clean_text[split_at:].lstrip()

    for i, chunk in enumerate(chunks):
        header = f"<b>🗞 Napi Globális Mélyelemzés ({i+1}/{len(chunks)})</b>\n\n"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": header + chunk, "parse_mode": "HTML"}
        requests.post(url, data=payload)
        time.sleep(1)

def ai_call(prompt, use_search=False):
    config = {}
    if use_search:
        config = {'tools': [{'google_search': {}}]}
    
    # Kényszerített rövidítés
    system_instruction = "Max 4-5 rövid mondat. Csak a lényeg. Tilos a Markdown és a felesleges sallang."
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=system_instruction + "\n" + prompt,
            config=config
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI Hiba: {e}")
        return "SKIP"

def main():
    history = load_history()
    past_context = "\n".join([f"- {h['date']}: {h['title']}" for h in history[-15:]])
    
    full_message = ""
    new_history_entries = []
    rss_items = [] # Az RSS-be szánt elemek listája

    # --- 1. ÁLLANDÓ HAZAI TÉMÁK ---
    print("Állandó hazai témák...")
    perm_topics = [
        ("ÜZEMANYAGÁRAK", "Friss benzin/gázolaj árak Magyarországon és várható változások."),
        ("POLITIKAI HELYZET", "Legfrissebb párt népszerűségi adatok és hírek.")
    ]
    
    for title, desc in perm_topics:
        res = ai_call(f"Téma: {title}. Feladat: {desc}\nKontextus:\n{past_context}", use_search=True)
        if res != "SKIP":
            full_message += f"⭐ {title}\n{res}\n\n"
            new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": title})
            rss_items.append({"title": title, "summary": res})

    # --- 2. NEMZETKÖZI KITEKINTŐ ---
    print("Nemzetközi szemle...")
    intl_res = ai_call("Bloomberg, Reuters: Hungary economy & forint status.", use_search=True)
    if intl_res != "SKIP":
        full_message += f"🌍 NEMZETKÖZI KITEKINTŐ\n{intl_res}\n\n"
        new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": "Nemzetközi Kitekintő"})
        rss_items.append({"title": "Nemzetközi Kitekintő", "summary": intl_res})

    # --- 3. RSS HÍREK ---
    print("RSS feldolgozás...")
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    blacklist = ["foci", "bajnokság", "celeb", "horoszkóp", "bulvár", "recept"]
    avoid_words = []

    scored_news = []
    for entry in feed.entries:
        if any(w in entry.title.lower() for w in blacklist + avoid_words): continue
        score = len(re.findall(r'<li>', entry.summary))
        scored_news.append({"title": entry.title, "summary": entry.summary, "score": score})

    for news in sorted(scored_news, key=lambda x: x['score'], reverse=True)[:5]:
        res = ai_call(f"Hír: {news['title']}\nForrás: {news['summary']}\nElemezd röviden.")
        if res != "SKIP":
            full_message += f"📌 {news['title'].upper()}\n{res}\n\n"
            new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": news['title']})
            rss_items.append({"title": news['title'], "summary": res})
        time.sleep(1)

    if full_message:
        send_telegram_chunked(full_message)
        save_history(history, new_history_entries)
        generate_rss(rss_items) # RSS fájl létrehozása
        print("Kész.")

if __name__ == "__main__":
    main()
