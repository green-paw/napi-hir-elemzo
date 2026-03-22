# rss_engine.py
import html
import re
from datetime import datetime, timedelta
from time import mktime
from typing import List, Dict, Tuple, Any, Set
import feedparser # type: ignore

from models import Article

def clean_news_text(entry: Any, field: str = 'title') -> str:
    """Kinyeri és megtisztítja a szöveget (HTML mentesítés, unescape)."""
    raw: str = entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))
    if not raw:
        return ""
    # HTML entitások feloldása és tagek törlése
    clean: str = re.sub(r'<[^>]+?>', '', html.unescape(raw))
    # Felesleges whitespace-ek eltávolítása
    return " ".join(clean.split()).strip()

def smart_truncate(text: str, max_length: int = 600) -> str:
    """Szóköz mentén vágja le a szöveget, ha túl hosszú."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def fetch_all_news(feeds_dict: Dict[str, Tuple[str, str]]) -> List[Article]:
    """Lekéri az összes RSS feedet, szűr, és Article objektumokat ad vissza."""
    news_pool: List[Article] = []
    seen_links: Set[str] = set()
    item_id: int = 0
    now: datetime = datetime.now()
    limit: timedelta = timedelta(hours=24)
    
    BLACKLIST: List[str] = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept"]

    for name, (url, description) in feeds_dict.items():
        print(f"📡 Letöltés: {name}...")
        try:
            feed: Any = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link in seen_links:
                    continue
                
                title: str = clean_news_text(entry, 'title')
                if not title:
                    continue
                    
                # Dátum parsing
                dt: datetime = now
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime.fromtimestamp(mktime(entry.published_parsed))
                    
                if now - dt > limit:
                    continue

                # Címkék kinyerése
                tags: List[str] = []
                if hasattr(entry, 'tags'):
                    tags = [t.term.lower() for t in entry.tags if hasattr(t, 'term')]
                    
                # Blacklist szűrés
                if any(b in title.lower() or b in tags for b in BLACKLIST):
                    continue

                # Leírás (summary) kinyerése
                raw_summary: str = entry.get('summary', entry.get('description', ''))
                summary: str = smart_truncate(clean_news_text({'summary': raw_summary}, 'summary'), 600)

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
            
    # Duplikátumok szűrése cím alapján is
    seen_titles: Set[str] = set()
    unique_news: List[Article] = []
    for n in news_pool:
        clean_title: str = n.title.strip().lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_news.append(n)
    
    print(f"🧹 Duplikátumok kiszűrve: {len(news_pool)} -> {len(unique_news)} hír.")
    
    # Időrendbe állítás (legrégebbi elöl, hogy az LLM lássa az ok-okozatot)
    unique_news.sort(key=lambda x: x.published)
    
    # ID-k újraosztása a rendezés után, hogy folytonos legyen a lista a klaszterezéshez
    for i, n in enumerate(unique_news):
        n.id = i
        
    return unique_news