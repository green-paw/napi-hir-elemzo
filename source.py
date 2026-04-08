import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from checkpoint_manager import NewsCache, load_checkpoint, save_checkpoint
import gemini_core
from pydantic import BaseModel, Field, field_validator, model_validator
import requests
from dataclasses import field
import time
import html
import re
import feedparser
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import config 
import hashlib


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

def generate_news_hash(title: str, link: str) -> str:
    """Stabil SHA-256 hasht generál a cím és a tisztított link alapján."""
    # Link tisztítása (query paraméterek nélkül a stabilitásért)
    clean_link = link.split('?')[0].split('#')[0].strip().lower()
    # Cím normalizálása
    clean_title = title.strip().lower()
    
    hash_base = f"{clean_title}|{clean_link}"
    return hashlib.sha256(hash_base.encode('utf-8')).hexdigest()

# --- MODELLEK ---


class NewsItem(BaseModel):
    id: str = Field(description="LLM-barát azonosító (pl. C1)")
    hash: str = "" # Alapértelmezésben üres, a validator tölti ki
    source_id: str
    source_meta: config.RssSource 
    link: str
    published: datetime
    title: str
    content: str
    embedding: Optional[List[float]] = None
    clean_content: Optional[str] = Field(default=None, exclude=True) # Ezt ellenőrizd!
    profile: Dict[str, float] = Field(default_factory=dict)

    @field_validator('title', 'content')
    @classmethod
    def validate_cleantext(cls, v: str) -> str:
        return cleantext(v)

    @model_validator(mode='after')
    def compute_hash(self) -> 'NewsItem':
        """Inicializálás után legenerálja a hash-t, ha még nincs."""
        if not self.hash:
            self.hash = generate_news_hash(self.title, self.link)
        return self

    def short_text_for_prompt(self, width: int = 500) -> str:
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
            if any(bad in tags for bad in BLACKLIST) or any(f"[{bad}]" in title_lower for bad in BLACKLIST): continue

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


def handle_news_feed_and_cache(incoming_news: List[NewsItem], run_id: str) -> Tuple[List[NewsItem], NewsCache]:
    cache_obj = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    
    full_blacklist: Set[str] = set().union(*cache_obj.trash_bin.values())
    existing_hashes: Set[str] = set().union(*(batch.keys() for batch in cache_obj.batches.values()))
    
    if run_id not in cache_obj.batches:
        cache_obj.batches[run_id] = {}

    for item in incoming_news:
        if item.hash in full_blacklist or item.hash in existing_hashes:
            continue
        cache_obj.batches[run_id][item.hash] = item

    # Takarítás
    limit = datetime.now() - timedelta(hours=24)
    cache_obj.batches = {ts: b for ts, b in cache_obj.batches.items() if datetime.fromisoformat(ts) > limit}
    cache_obj.trash_bin = {ts: t for ts, t in cache_obj.trash_bin.items() if datetime.fromisoformat(ts) > limit}

    # Első mentés (új hírek hash-ei megvannak)
    save_checkpoint("news_feed.json", cache_obj, NewsCache)
    
    all_live_news = []
    for batch in cache_obj.batches.values():
        all_live_news.extend(batch.values())
        
    return all_live_news, cache_obj

def update_current_batch(items: List[NewsItem], cache: NewsCache, run_id: str):
    if run_id not in cache.batches:
        cache.batches[run_id] = {}
    for item in items:
        cache.batches[run_id][item.hash] = item
    save_checkpoint("news_feed.json", cache, NewsCache)

def add_to_trash(item_hash: str, cache: NewsCache, run_id: str):
    if run_id not in cache.trash_bin:
        cache.trash_bin[run_id] = set()
    cache.trash_bin[run_id].add(item_hash)

    for tid in list(cache.batches.keys()):
        cache.batches[tid].pop(item_hash, None)
        if not cache.batches[tid]:
            del cache.batches[tid]

def embed_news(all_live_news: List[NewsItem], cache_obj: NewsCache, RUN_ID: str) -> None:
    news_to_embed = [
        item for item in all_live_news 
        if item.embedding is None and item.clean_content is not None
    ]

    if news_to_embed:
        print(f"🧬 {len(news_to_embed)} hír embeddingjének lekérése...")
        texts = [str(item.clean_content) for item in news_to_embed]
        new_vectors = gemini_core.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        for item, vector in zip(news_to_embed, new_vectors):
            item.embedding = vector
        update_current_batch(all_live_news, cache_obj, RUN_ID)

def score_items(items: List[NewsItem], anchor_vectors: Dict[str, np.ndarray]):
    """Kitölti az item.profile-t az ANCHORS alapján."""
    if not items:
        return

    # Kigyűjtjük az embeddingeket egy nagy mátrixba (N hír x D dimenzió)
    news_embeddings = np.array([item.embedding for item in items if item.embedding is not None])
    
    # Kigyűjtjük a horgonyokat (M horgony x D dimenzió)
    anchor_names = list(anchor_vectors.keys())
    # anchor_vectors[name] nálad (1, -1) alakú, ezért flatten() vagy reshape kell
    anchor_matrix = np.vstack([anchor_vectors[name] for name in anchor_names])

    # Kiszámoljuk a hasonlóságot minden hír és minden horgony között
    # Eredmény: (N hír x M horgony) mátrix
    similarities = cosine_similarity(news_embeddings, anchor_matrix)

    # Visszaírjuk a profilokba
    for i, item in enumerate(items):
        for j, name in enumerate(anchor_names):
            item.profile[name] = float(similarities[i, j])

def cluster_news(items: List[NewsItem], threshold: float = 0.3):
    """Csoportosítja a híreket hasonlóság alapján."""
    embeddings = np.array([item.embedding for item in items if item.embedding is not None])
    
    if len(embeddings) < 2:
        return

    # AgglomerativeClustering koszinusz távolsággal
    # Megjegyzés: A koszinusz távolság = 1 - koszinusz hasonlóság
    model = AgglomerativeClustering(
        n_clusters=None, 
        metric='cosine', 
        linkage='average',
        distance_threshold=threshold 
    )
    
    labels = model.fit_predict(embeddings)
    
    # Labels hozzárendelése (pl. item.cluster_id mezőbe)
    for i, item in enumerate(items):
        item.profile["cluster_id"] = int(labels[i])