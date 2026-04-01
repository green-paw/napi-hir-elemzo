import time
import html
import re
import feedparser
import textwrap
from datetime import datetime, timedelta
from typing import List, Optional, Set

from pydantic import BaseModel, Field, field_validator
# Feltételezve, hogy a config-ban már az új RssSource struktúra van
import config 
from checkpoint_manager import load_checkpoint, save_checkpoint

# --- SEGÉDFÜGGVÉNYEK ---

def cleantext(raw: str) -> str:
    """HTML mentesítés, entitás dekódolás és whitespace normalizálás."""
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    # Tagek cseréje szóközre (hogy ne ragadjanak össze a szavak)
    no_html = re.sub(r'<[^>]+?>', ' ', unescaped)
    return " ".join(no_html.split()).strip()

def extract_safe_text(entry, field: str) -> str:
    """Biztonságos adatkinyerés feedparser entry-ből."""
    if field == 'content':
        if 'content' in entry and isinstance(entry.content, list) and len(entry.content) > 0:
            return entry.content[0].get('value', '')
        return entry.get('summary_detail', {}).get('value', entry.get('summary', ''))
    
    return entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))

# --- MODELLEK ---

class NewsItem(BaseModel):
    id: str = Field(description="LLM-barát azonosító (pl. C1)")
    source_id: str
    source_meta: config.RssSource # A config-ban definiált dataclass
    link: str
    published: datetime
    title: str
    content: str
    embedding: Optional[List[float]] = None

    @field_validator('title', 'content')
    @classmethod
    def validate_cleantext(cls, v: str) -> str:
        return cleantext(v)

    def short_text_for_prompt(self, width: int = 500) -> str:
        """Költséghatékony szöveg az LLM-nek."""
        combined = f"{self.title} - {self.content}"
        return textwrap.shorten(combined, width=width, placeholder="...")

# --- FŐ FÜGGVÉNY ---

def fetch_news() -> List[NewsItem]:
    # Cache/Checkpoint ellenőrzése
    news_pool: List[NewsItem] = load_checkpoint("news_pool.json", List[NewsItem]) or []
    if news_pool:
        print("📦 Hírek betöltve a checkpointból.")
        return news_pool

    seen_links: Set[str] = set()
    now = datetime.now()
    limit = timedelta(hours=24)
    
    BLACKLIST = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept", "életmód", "bulvar", "tv-műsor"]

    print(f"📰 Hírek lekérése ({limit.days * 24}h limit)...")
    
    temp_pool: List[NewsItem] = []

    for name, source in config.RSS_SOURCES.items():
        try:
            feed = feedparser.parse(source.url)
            for entry in feed.entries:
                # 0. Duplikáció szűrés link alapján
                if entry.link in seen_links:
                    continue

                # 1. Időbeli szűrés
                dt = now
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                
                if now - dt > limit:
                    continue
                
                # 2. Kategória szűrés (Blacklist)
                tags = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
                title_lower = entry.title.lower()
                if any(bad in tags for bad in BLACKLIST) or any(f"[{bad}]" in title_lower for bad in BLACKLIST):
                    continue

                # 3. Adatkinyerés
                raw_title = extract_safe_text(entry, 'title')
                raw_content = extract_safe_text(entry, 'content')
                
                if not raw_title:
                    continue

                # 4. Objektum létrehozása (Az automatikus tisztítás itt fut le!)
                item = NewsItem(
                    id="",
                    source_id=name,
                    source_meta=source,
                    link=entry.link,
                    published=dt,
                    title=raw_title,
                    content=raw_content
                )
                
                temp_pool.append(item)
                seen_links.add(entry.link)
                
        except Exception as e:
            print(f"⚠️ Hiba a(z) {name} forrásnál: {e}")
            
    # 5. Cím alapú duplikáció szűrés (tisztított címekkel)
    seen_titles = set()
    unique_news: List[NewsItem] = []

    for n in temp_pool:
        # A Pydantic már megtisztította a n.title-t
        clean_t = n.title.strip().lower()
        if clean_t not in seen_titles:
            seen_titles.add(clean_t)
            unique_news.append(n)
    
    # 6. Időrendbe tétel és ID-k újrakiosztása (hogy ne legyenek lyukak a szűrés után)
    unique_news.sort(key=lambda x: x.published, reverse=True)
    
    final_pool: List[NewsItem] = []
    for idx, item in enumerate(unique_news):
        item.id = f"C{idx + 1}" # Újrageneráljuk a sorrend miatt
        final_pool.append(item)

    print(f"✅ Begyűjtés kész: {len(final_pool)} egyedi, releváns hír.")

    save_checkpoint("news_pool.json", final_pool, List[NewsItem])
    return final_pool