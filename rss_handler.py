import time
import html
import re
import feedparser
import config
from datetime import datetime, timedelta

def clean_news_text(entry, field='title'):
    """Kinyeri és megtisztítja a szöveget (HTML mentesítés, unescape)."""
    raw = entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))
    if not raw:
        return ""
    # HTML entitások feloldása és tagek törlése
    clean = re.sub(r'<[^>]+?>', '', html.unescape(raw))
    # Felesleges whitespace-ek eltávolítása
    return " ".join(clean.split()).strip()

def smart_truncate(text, max_length=600):
    """Szóköz mentén vágja le a szöveget, ha túl hosszú."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def fetch_news():
    """Begyűjti, szűri és tisztítja a híreket az összes RSS forrásból."""
    news_pool = []
    item_id = 0
    now = datetime.now()
    limit = timedelta(hours=24)
    
    # Stratégiai zajszűrő feketelista
    BLACKLIST = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept", "életmód", "bulvar"]

    print(f"📰 Hírek lekérése és szűrése ({limit.days * 24}h limit)...")
    
    for name, url in config.RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # 1. IDŐBELI SZŰRÉS (Októberi hírek és elavult tartalom ellen)
                dt = now
                if hasattr(entry, 'published_parsed'):
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if now - dt > limit:
                        continue
                
                # 2. KATEGÓRIA SZŰRÉS (Magyar tags/címkék alapján)
                tags = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
                if any(bad in tags for bad in BLACKLIST):
                    continue

                # 3. TISZTÍTÁS
                title = clean_news_text(entry, 'title')
                if not title:
                    continue

                news_pool.append({
                    "id": item_id,
                    "source": name,
                    "title": title,
                    "summary": smart_truncate(clean_news_text(entry, 'summary'), 600),
                    "link": entry.link,
                    "tags": tags,
                    "published": dt
                })
                item_id += 1
        except Exception as e:
            print(f"⚠️ Hiba a(z) {name} forrásnál: {e}")
            
    print(f"✅ Begyűjtés kész: {len(news_pool)} releváns hír.")
    return news_pool
