import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from checkpoint_manager import load_checkpoint, save_checkpoint
import gemini_core
import requests
import time
import feedparser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import config 

from models import NewsCache, NewsItem

# --- SEGÉDFÜGGVÉNYEK ---

def extract_safe_text(entry, field: str) -> str:
    """Biztonságos adatkinyerés feedparser entry-ből."""
    if field == 'content':
        if 'content' in entry and isinstance(entry.content, list) and len(entry.content) > 0:
            return entry.content[0].get('value', '')
        return entry.get('summary_detail', {}).get('value', entry.get('summary', ''))
    
    return entry.get(f"{field}_detail", {}).get('value', entry.get(field, ''))


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

    for item in unique_news:
        if not item.hash:
            item.compute_hash()

    return unique_news


def handle_news_feed_and_cache(incoming_news: List[NewsItem], run_id: str) -> Tuple[List[NewsItem], NewsCache]:
    for item in incoming_news:
        if not item.hash:
            item.compute_hash()

    cache_obj = NewsCache()
    #cache_obj = load_checkpoint("news_feed.json", NewsCache) or NewsCache()
    
    full_blacklist: Set[str] = set().union(*cache_obj.trash_bin.values())
    existing_hashes: Set[str] = set().union(*(batch.keys() for batch in cache_obj.batches.values()))
    
    if run_id not in cache_obj.batches:
        cache_obj.batches[run_id] = {}

    for item in incoming_news:
        if item.hash in full_blacklist or item.hash in existing_hashes:
            continue
        if any(item.hash in batch for batch in cache_obj.batches.values()):
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

ANCHORS = {
    "POL": (
        "politika, pártpolitika, kormány, ellenzék, választás, kampány, parlament, "
        "szavazás, diplomácia, közpolitika, törvényhozás, külpolitika, belpolitika, "
        "politics, government, elections, parliament, voting, diplomacy, policy, "
        "legislation, state, political party"
    ),
    "ECO": (
        "gazdaság, pénzügy, tőzsde, infláció, GDP, költségvetés, adózás, bankrendszer, "
        "befektetés, makrogazdaság, kamatláb, valutapiac, economy, finance, stock market, "
        "inflation, budget, taxation, banking, investment, macroeconomics, interest rates, currency"
    ),
    "TEC": (
        "számítástechnika, szoftver, hardver, mesterséges intelligencia, MI, AI, "
        "kiberbiztonság, programozás, félvezető, GPU, felhő alapú, digitalizáció, "
        "robotika, kódolás, technology, software, hardware, artificial intelligence, "
        "cybersecurity, programming, semiconductor, robotics, coding, cloud computing"
    ),
    "HUN": (
        "Magyarország, Budapest, magyar, belföld, hazai, forint, tiszapárt, fidesz, "
        "magyar kormány, magyar hír, Hungary, Hungarian, Budapest, forint, HUF, "
        "local news Hungary, Hungarian government"
    ),
    "TRASH": (
        "bulvár, pletyka, celeb, horoszkóp, társkereső, profil, társkeresés, szex, "
        "kattintásvadász, botrány, életmód, wellness, recept, főzés, gossip, "
        "celebrity, horoscope, dating, profile, dating site, clickbait, scandal, "
        "lifestyle, recipe, cooking, entertainment"
    )
}

def get_anchor_embeddings() -> Dict[str, np.ndarray]:
    anchor_cache_file = "anchors.json"
    cached = load_checkpoint(anchor_cache_file, Dict[str, List[float]])
    if cached:
        return {k: np.array(v).reshape(1, -1) for k, v in cached.items()}

    print("⚓ Többirányú horgony-vektorok generálása...")
    keys = list(ANCHORS.keys())
    texts = list(ANCHORS.values())
    vectors = gemini_core.embed(texts, task_type="RETRIEVAL_QUERY")
    
    anchor_dict = {keys[i]: vectors[i] for i in range(len(keys))}
    save_checkpoint(anchor_cache_file, anchor_dict, Dict[str, List[float]])
    return {k: np.array(v).reshape(1, -1) for k, v in anchor_dict.items()}





from datetime import datetime

def deduplicate_to_chronological_batches(cache_obj: NewsCache) -> NewsCache:
    """
    Deduplikálja a híreket, majd a publikálási időpontjuk alapján 
    besorolja őket a legmegfelelőbb időrendi batch-be.
    """
    # 1. Összes elem kigyűjtése és alap-deduplikáció (legkorábbi publikálás tartunk meg)
    all_items = []
    for batch in cache_obj.batches.values():
        all_items.extend(batch.values())
    
    if not all_items:
        return cache_obj

    # Rendszerezés: hash szerint csak a legelsőt tartjuk meg
    # (Előtte sorbarendezzük, hogy biztosan a legkorábbi példány legyen az első)
    all_items.sort(key=lambda x: x.published)
    unique_items: Dict[str, NewsItem] = {}
    for item in all_items:
        if item.hash not in unique_items:
            unique_items[item.hash] = item

    # 2. Batch kulcsok (időbélyegek) előkészítése
    # ISO formátumú stringeket datetime-má alakítjuk a hasonlításhoz
    sorted_batch_times = []
    for ts_str in cache_obj.batches.keys():
        try:
            sorted_batch_times.append((datetime.fromisoformat(ts_str), ts_str))
        except ValueError:
            continue
    
    # Időrendbe rakjuk a batch-időpontokat
    sorted_batch_times.sort() 

    # 3. Új struktúra felépítése
    new_batches: Dict[str, Dict[str, NewsItem]] = {ts_str: {} for _, ts_str in sorted_batch_times}
    
    # Ha van olyan hír, ami régebbi, mint a legelső batch-ünk, 
    # azt is bele kell tennünk valahova (pl. a legelsőbe)
    fallback_batch_str = sorted_batch_times[0][1] if sorted_batch_times else None

    for item in unique_items.values():
        assigned = False
        # Megkeressük az első olyan batch-et, ami a hír publikálása UTÁN jött létre
        for batch_dt, batch_str in sorted_batch_times:
            if item.published <= batch_dt:
                new_batches[batch_str][item.hash] = item
                assigned = True
                break
        
        # Ha a hír újabb, mint az eddigi összes batch, megy a legutolsóba
        if not assigned and fallback_batch_str:
            last_batch_str = sorted_batch_times[-1][1]
            new_batches[last_batch_str][item.hash] = item

    # 4. Üresen maradt batchek takarítása (opcionális)
    cache_obj.batches = {k: v for k, v in new_batches.items() if v}
    
    return cache_obj