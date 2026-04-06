import requests
from dataclasses import field
import time
import html
import re
import feedparser
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator
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
    profile: Dict[str, float] = field(default_factory=dict)

    @field_validator('title', 'content')
    @classmethod
    def validate_cleantext(cls, v: str) -> str:
        return cleantext(v)

    def short_text_for_prompt(self, width: int = 500) -> str:
        """Költséghatékony szöveg az LLM-nek."""
        combined = f"{self.title} - {self.content}"
        return textwrap.shorten(combined, width=width, placeholder="...")

# --- FŐ FÜGGVÉNY ---

# --- SZÁLKEZELT MUNKAFÜGGVÉNY ---

def process_single_source(args: Tuple[str, config.RssSource, datetime, timedelta]) -> List[NewsItem]:
    """Egyetlen RSS forrás feldolgozása egy külön szálon."""
    name, source, now, limit = args
    BLACKLIST: List[str] = ["sport", "bulvár", "szórakozás", "horoszkóp", "időjárás", "recept", "életmód", "bulvar", "tv-műsor"]
    local_items: List[NewsItem] = []

    try:
        response = requests.get(source.url, timeout=(5, 15))
        response.raise_for_status() # Hiba esetén (pl. 404) kivételt dob
        feed = feedparser.parse(response.content)
        for entry in feed.entries:

            dt: datetime = now
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
            
            if now - dt > limit:
                continue
            
            # 2. Kategória szűrés
            tags: List[str] = [t.term.lower() for t in entry.get('tags', []) if hasattr(t, 'term')]
            title_lower: str = entry.title.lower()
            if any(bad in tags for bad in BLACKLIST) or any(f"[{bad}]" in title_lower for bad in BLACKLIST):
                continue

            # 3. Adatkinyerés
            raw_title: str = extract_safe_text(entry, 'title')
            raw_content: str = extract_safe_text(entry, 'content')
            
            if not raw_title:
                continue

            item = NewsItem(
                id="",
                source_id=name,
                source_meta=source,
                link=entry.link,
                published=dt,
                title=raw_title,
                content=raw_content
            )
            local_items.append(item)
            
    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout: {source.url} nem válaszolt időben.")
        return []
    except Exception as e:
        print(f"❌ Hiba a forrásnál ({source.url}): {e}")
        return []
    
    return local_items

# --- FŐ FÜGGVÉNY ---

def fetch_news() -> List[NewsItem]:
    now: datetime = datetime.now()
    limit: timedelta = timedelta(hours=24)
    
    print(f"📰 Hírek lekérése ({limit.days * 24}h limit) threading használatával...")

    # Előkészítjük a feladatokat a ThreadPool-nak
    tasks: List[Tuple[str, config.RssSource, datetime, timedelta]] = [
        (name, source, now, limit) for name, source in config.RSS_SOURCES.items()
    ]

    temp_pool: List[NewsItem] = []
    
    # Max szálak száma: források száma vagy egy ésszerű limit (pl. 10)
    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
        results = executor.map(process_single_source, tasks)
        for result_list in results:
            temp_pool.extend(result_list)

    # Utófeldolgozás: Duplikáció szűrés (Link és Cím)
    seen_links: Set[str] = set()
    seen_titles: Set[str] = set()
    unique_news: List[NewsItem] = []

    # Időrendbe rakjuk először, hogy a legfrissebbek maradjanak meg duplikáció esetén
    temp_pool.sort(key=lambda x: x.published, reverse=True)

    for n in temp_pool:
        clean_t: str = n.title.strip().lower()
        if n.link not in seen_links and clean_t not in seen_titles:
            seen_links.add(n.link)
            seen_titles.add(clean_t)
            unique_news.append(n)
    
    # ID-k kiosztása
    for idx, item in enumerate(unique_news):
        item.id = f"C{idx + 1}"

    print(f"✅ Begyűjtés kész: {len(unique_news)} egyedi hír.")
    return unique_news