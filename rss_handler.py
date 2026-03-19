import time
import html
import re
import feedparser
import config
from datetime import datetime, timedelta
from models import Article # Ne felejtsd el az importot!

def clean_news_text(entry, field='title'):
    raw = entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))
    if not raw:
        return ""
    clean = re.sub(r'<[^>]+?>', '', html.unescape(raw))
    return " ".join(clean.split()).strip()

def smart_truncate(text, max_length=600):
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def fetch_news():
    news_pool = [] # Ezt a nevet használjuk végig
    seen_links = set()
    item_id = 0
    now = datetime.now()
    limit = timedelta(hours=24)
    
    BLACKLIST = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept", "életmód", "bulvar", "tv-műsor"]

    print(f"📰 Hírek lekérése és szűrése ({limit.days * 24}h limit)...")
    
    for name, source_data in config.RSS_SOURCES.items():
        try:
            url = source_data[0]
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link in seen_links:
                    continue

                # 1. IDŐBELI SZŰRÉS
                dt = now
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                
                if now - dt > limit:
                    continue
                
                # 2. KATEGÓRIA SZŰRÉS
                tags = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
                title_lower = entry.get('title', '').lower()
                if any(bad in tags for bad in BLACKLIST) or any(f"[{bad}]" in title_lower for bad in BLACKLIST):
                    continue

                # 3. TISZTÍTÁS ÉS OBJEKTUM LÉTREHOZÁS
                title = clean_news_text(entry, 'title')
                if not title:
                    continue

                raw_summary = entry.get('summary', entry.get('description', ''))
                summary = smart_truncate(clean_news_text({'summary': raw_summary}, 'summary'), 600)

                # Itt példányosítjuk az Article objektumot
                news_pool.append(Article(
                    id=item_id,
                    title=title,
                    link=entry.link,
                    source=name,
                    summary=summary,
                    content=summary,
                    published=dt  # <--- Itt a 'published' nevet használd
                ))
                
                seen_links.add(entry.link)
                item_id += 1
                
        except Exception as e:
            print(f"⚠️ Hiba a(z) {name} forrásnál: {e}")
    
    # Rendezés az Article objektum 'date' mezője alapján
    news_pool.sort(key=lambda x: x.published, reverse=True)
    
    print(f"✅ Begyűjtés kész: {len(news_pool)} egyedi, releváns hír.")
    return news_pool
