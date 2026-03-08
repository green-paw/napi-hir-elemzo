import os
import feedparser
import re
import time
import json
import requests
from google import genai
from datetime import datetime

# --- KONFIGURÁCIÓ ---
# Fontos: Ellenőrizd, hogy a GitHub Secrets-ben GOOGLE_API_PAID_KEY néven van-e a kulcs!
GOOGLE_API_KEY = os.getenv("GOOGLE_API_PAID_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"
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
    combined = history + new_entries
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(combined[-40:], f, ensure_ascii=False, indent=2)

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
    """Általános hívás a modellhez, opcionális Google kereséssel."""
    config = {}
    if use_search:
        config = {'tools': [{'google_search': {}}]}
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
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

    # --- 1. ÁLLANDÓ HAZAI TÉMÁK (Grounding/Search használatával) ---
    print("Állandó hazai témák elemzése...")
    perm_topics = [
        ("ÜZEMANYAGÁRAK", "Friss benzin/gázolaj árak Magyarországon, várható változások és olajár/forint hatás."),
        ("POLITIKAI HELYZET", "Legfrissebb közvélemény-kutatások, pártok népszerűsége és 2026-os választási kilátások.")
    ]
    
    for title, desc in perm_topics:
        prompt = f"Elemezd: {title}. Feladat: {desc}\nElőzmények:\n{past_context}\nSZABÁLY: Száraz tények, nulla feltételezés, nulla Markdown."
        analysis = ai_call(prompt, use_search=True)
        if analysis != "SKIP":
            full_message += f"⭐ {title}\n{analysis}\n\n"
            new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": title})

    # --- 2. NEMZETKÖZI KITEKINTŐ (Bloomberg, Reuters, FT) ---
    print("Nemzetközi sajtó szemlézése...")
    intl_prompt = f"""
    Keress rá angolul: Bloomberg, Reuters, Financial Times 'Hungary economy', 'Forint'. 
    Foglald össze a nemzetközi befektetői hangulatot és a magyar gazdaság megítélését.
    Előzmények:\n{past_context}\nSZABÁLY: Száraz tények, nulla Markdown.
    """
    intl_analysis = ai_call(intl_prompt, use_search=True)
    if intl_analysis != "SKIP":
        full_message += f"🌍 NEMZETKÖZI KITEKINTŐ (Bloomberg, Reuters)\n{intl_analysis}\n\n"
        new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": "Nemzetközi Kitekintő"})

    # --- 3. RSS HÍREK ELEMZÉSE ---
    print("RSS hírek feldolgozása...")
    feed = feedparser.parse("https://news.google.com/rss?hl=hu&gl=HU&ceid=HU:hu")
    blacklist = ["foci", "bajnokság", "celeb", "horoszkóp", "bulvár", "recept"]
    avoid_words = ["benzin", "gázolaj", "üzemanyag", "választás", "közvélemény-kutatás"]

    scored_news = []
    for entry in feed.entries:
        if any(w in entry.title.lower() for w in blacklist + avoid_words): continue
        score = len(re.findall(r'<li>', entry.summary))
        scored_news.append({"title": entry.title, "summary": entry.summary, "score": score})

    for news in sorted(scored_news, key=lambda x: x['score'], reverse=True)[:5]:
        prompt = f"Elemezd ezt a hírt: {news['title']}\nForrás: {news['summary']}\nElőzmények:\n{past_context}\nSZABÁLY: Mutasd be a kormányközeli és kritikus narratívát, a gazdasági hatást és a tényeket. Nulla Markdown."
        analysis = ai_call(prompt)
        if analysis != "SKIP":
            full_message += f"📌 {news['title'].upper()}\n{analysis}\n\n"
            new_history_entries.append({"date": datetime.now().strftime("%m.%d %H:%M"), "title": news['title']})
        time.sleep(1)

    if full_message:
        send_telegram_chunked(full_message)
        save_history(history, new_history_entries)
        print("Kész.")

if __name__ == "__main__":
    main()
