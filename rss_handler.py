import time
import html
import re
import feedparser
import config
from datetime import datetime, timedelta

from typing import List
from models import Article

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

def fetch_news() -> List[Article]:
    news_pool: List[Article] = []
    seen_links = set() # Duplikáció szűréshez
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
                # 0. DUPLIKÁCIÓ SZŰRÉS
                if entry.link in seen_links:
                    continue

                # 1. IDŐBELI SZŰRÉS
                dt = now
                if hasattr(entry, 'published_parsed'):
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                elif hasattr(entry, 'updated_parsed'): # Tartalék, ha nincs published
                    dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                
                if now - dt > limit:
                    continue
                
                # 2. KATEGÓRIA SZŰRÉS
                tags = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
                # Nézzük meg a címet is, mert sokszor oda írják a rovatot (pl. [Sport])
                title_lower = entry.title.lower()
                if any(bad in tags for bad in BLACKLIST) or any(f"[{bad}]" in title_lower for bad in BLACKLIST):
                    continue

                # 3. TISZTÍTÁS ÉS ÖSSZEGYŰJTÉS
                title = clean_news_text(entry, 'title')
                if not title:
                    continue

                # Próbáljuk kinyerni a leírást több mezőből is
                raw_summary = entry.get('summary', entry.get('description', ''))
                summary = smart_truncate(clean_news_text({'summary': raw_summary}, 'summary'), 600)

                news_pool.append(Article(
                    id=item_id,
                    source=name,
                    title=title,
                    summary=summary,
                    link=entry.link,
                    tags=tags,
                    published=dt
                ))
                
                seen_links.add(entry.link)
                item_id += 1
                
        except Exception as e:
            print(f"⚠️ Hiba a(z) {name} forrásnál: {e}")
            
    seen_titles = set()
    unique_news = []
    for n in news_pool:
        # Tisztítjuk a címet (kisbetű, szóközök le) a pontosabb egyezésért
        clean_title = n.title.strip().lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_news.append(n)
    
    print(f"🧹 Duplikátumok kiszűrve: {len(news_pool)} -> {len(unique_news)} hír.")
    news_pool = unique_news # Ezzel dolgozunk tovább

    # Időrendbe tétel (legfrissebb elöl), hogy a main() fixen a legújabbakat lássa
    news_pool.sort(key=lambda x: x.published, reverse=True)
    
    print(f"✅ Begyűjtés kész: {len(news_pool)} egyedi, releváns hír.")
    return news_pool
