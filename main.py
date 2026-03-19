# saját importok
import config
import output_handler

# Csak a szükséges handler funkciók
from gemini_handler import (
    get_strategic_topics, validate_news_clusters, 
    generate_event_summary, get_gemini_embeddings, 
    translate_if_needed, MultiClusterResponse, refine_event_list,
    validate_news_clusters_batch
)
from models import Article, ArticleSource, FinalEvent

from rss_handler import fetch_news

# általános importok
import math
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor 
from sklearn.cluster import AgglomerativeClustering

from datetime import datetime
import random

def myPrint(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# --- Szemantikus szűrő matematikai alapjai ---
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Két vektor koszinusz-hasonlóságának kiszámítása."""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0

def calculate_priority_score(event) -> int:
    """
    Kiszámolja a végleges 0-100-as prioritási pontszámot.
    Itt az 'event' a gemini_handler-ből visszakapott ClusterResultSingle objektum.
    """
    if not event or not hasattr(event, 'scores'):
        return 0

    # Pydantic objektum attribútumainak elérése pont-notációval
    s = event.scores
    llm_base = (s.impact * 0.6) + (s.relevance * 0.3) + (s.novelty * 0.1)
    
    cluster_size = len(event.ids)
    size_bonus = min(cluster_size * 0.8, 25.0) 
    final_score = (llm_base * 7.5) + size_bonus
    
    return min(round(final_score), 100)
    
def semantic_filter(news_pool: List[Article], topics: List[str], top_p=0.85) -> List[Article]:
    """
    Szemantikus rangsorolás és szűrés. 
    Bemenet és kimenet is Article objektumok listája.
    """
    if not topics or not news_pool: 
        return news_pool
        
    myPrint(f"🔍 Szemantikus rangsorolás: {len(news_pool)} hír...")
    
    topic_embs = get_gemini_embeddings(topics)
    
    # 1. SZUPER-SZÖVEG generálása az objektum mezőiből
    # Feltételezve, hogy az Article-ben a content vagy summary mezőben van a lényeg
    news_texts = [
        f"CÍM: {n.title} FORRÁS: {n.source} TARTALOM: {n.content[:200]}" 
        for n in news_pool
    ]
    news_embs = get_gemini_embeddings(news_texts)
    
    # 2. Relevancia számítása és VEKTOROK MENTÉSE az objektumba
    filtered_with_scores = []
    
    for i, n_emb in enumerate(news_embs):
        # Az embeddinget közvetlenül az Article objektumban tároljuk el
        news_pool[i].embedding = n_emb
        
        sims = [cosine_similarity(n_emb, t_emb) for t_emb in topic_embs]
        match_score = max(sims) if sims else 0
        
        # Csak azokat tartjuk meg, amik átmennek a küszöbön
        if match_score > top_p:
            # Ideiglenesen egy tuple-ben tároljuk a score-t a rendezéshez
            filtered_with_scores.append((news_pool[i], match_score))

    # Rendezés a match_score alapján (a tuple második eleme)
    filtered_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Csak az Article objektumokat adjuk vissza
    final_filtered = [item[0] for item in filtered_with_scores]
    
    avg_score = sum(score for _, score in filtered_with_scores) / len(filtered_with_scores) if filtered_with_scores else 0
    myPrint(f"✅ Rangsorolás kész: {len(final_filtered)} hír továbbküldve (átlagos relevancia: {avg_score:.2f})")
    
    return final_filtered

def cluster_news(news_pool: List[Article]):
    if not news_pool: 
        return [], []
    
    myPrint(f"🧩 Klaszterezés ({len(news_pool)} hír)...")
    
    # 1. VEKTOROK ÚJRAHASZNOSÍTÁSA (Pydantic objektumból)
    embeddings = [n.embedding for n in news_pool if n.embedding is not None]
    if not embeddings:
        myPrint("⚠️ Hiba: Hiányzó embeddingek a klaszterezéshez!")
        return [], []

    groups = auto_cluster(embeddings, news_pool)
    total_raw_groups = len(groups)

    clusters_to_validate = []
    discarded_summaries = []

    for label, news_list in groups.items():
        count = len(news_list)
        # Itt a match_score-t használjuk, amit a semantic_filter-ben számoltunk
        avg_relevance = sum(getattr(n, 'match_score', 0) for n in news_list) / count if count > 0 else 0

        summary_text = f"{news_list[0].title} ({count} hír)"
        
        # Előszűrési feltételek
        if count >= 3:
            clusters_to_validate.append(news_list)
        elif count == 2 and avg_relevance > 0.92:
            clusters_to_validate.append(news_list)
        elif count == 1 and avg_relevance > 0.97:
            clusters_to_validate.append(news_list)
        else:
            discarded_summaries.append(summary_text)
    
    num_to_process = len(clusters_to_validate)
    myPrint(f"📉 Szűrés után {num_to_process}/{total_raw_groups} klaszter maradt validálásra.")
    
    final_validated_events = []
    
    # 2. BATCH VALIDÁCIÓ (Pl. 5-ös csoportokban küldjük a Lite-nak)
    batch_size = 5
    for i in range(0, num_to_process, batch_size):
        current_batch = clusters_to_validate[i : i + batch_size]
        myPrint(f" 🚀 Lite Batch validáció: {i//batch_size + 1}. csoport ({len(current_batch)} klaszter)...")
        
        # A gemini_handler-ben a validate_news_clusters_batch-et kell hívni!
        # Ez a függvény a háttérben MultiClusterResponse-t ad vissza
        batch_result = validate_news_clusters_batch(current_batch)

        if batch_result and batch_result.events:
            for event in batch_result.events:
                # Mivel a Lite csak korlátozott cikkmennyiséget látott, 
                # itt visszakeressük az eredeti teljes ID listát a klaszterből
                # (A batch_result eseményeiben lévő ID-k alapján azonosítjuk a klasztert)
                
                # Ez a rész feltételezi, hogy a Lite visszaküldi az ID-kat is.
                # A pontszámításhoz elmentjük az eseményt
                final_validated_events.append(event)

    return final_validated_events, discarded_summaries

def auto_cluster(embeddings: List[List[float]], news_pool: List[Article]):
    distance_limit = 0.125
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_limit,
        metric='cosine', 
        linkage='complete' 
    ).fit(embeddings)
    
    groups = {}
    for idx, label in enumerate(clustering.labels_):
        groups.setdefault(label, []).append(news_pool[idx])
        
    max_size = max(len(g) for g in groups.values()) if groups else 0
    myPrint(f"✨ Optimális klaszterezés elérve ({len(groups)} csoport, legnagyobb: {max_size} hír).")
    
    return groups
    
def main():
    myPrint("🚀 Hírfigyelő rendszer indítása...")
    
    # 1. Lekérés (Article objektumok listáját kapjuk)
    raw_news: List[Article] = fetch_news()
    if not raw_news:
        myPrint("❌ Nincs bejövő hír, leállás.")
        return
    
    # --- Duplikátum szűrés ---
    seen_titles = set()
    unique_news: List[Article] = []
    for n in raw_news:
        clean_title = n.title.strip().lower()
        if clean_title not in seen_titles:
            seen_titles.add(clean_title)
            unique_news.append(n)
    
    myPrint(f"🧹 Duplikátumok eltávolítva: {len(raw_news)} -> {len(unique_news)} egyedi hír maradt.")
    
    # 2. Stratégiai témák generálása
    sample_size = min(len(unique_news), 300)
    # A random.sample már objektumokból válogat, n.title-t használunk
    titles_sample = "\n".join([n.title for n in random.sample(unique_news, sample_size)])
    topics = get_strategic_topics(titles_sample)
    
    if not topics:
        myPrint("⚠️ Nem sikerült stratégiai témákat generálni, leállás.")
        return

    myPrint("🎯 Napi Stratégiai Topikok:")
    for i, t in enumerate(topics, 1):
        myPrint(f"  {i}. {t}")
        
    topics_html = "<ul>" + "".join([f"<li>{t}</li>" for t in topics]) + "</ul>"
        
    # 3. Szemantikus szűrés (Article objektumokat küldünk és kapunk vissza)
    filtered_news = semantic_filter(unique_news, topics, top_p=0.86)
    if not filtered_news:
        myPrint("❌ A szemantikus szűrés után nem maradt hír, leállás.") 
        return

    # 4. Klaszterezés és Lite Validáció (Batching benne van a függvényben)
    # all_events most már ClusterResultSingle objektumok listája
    all_events, discarded_summaries = cluster_news(filtered_news)
    
    if not all_events:
        myPrint("❌ Nem sikerült eseményeket generálni a klaszterekből.")
        return

    # 5. Hibrid Pontozás és Sorbarendezés (Pydantic objektum attribútumokkal)
    myPrint(f"⚖️ Végleges pontszámok kiszámítása {len(all_events)} eseményre...")
    
    # Itt közvetlenül az objektumhoz adhatjuk a pontszámot, vagy használhatunk egy tuple-t
    scored_events = []
    for ev in all_events:
        score = calculate_priority_score(ev)
        scored_events.append((ev, score))
    
    # Rendezés a score (második elem) alapján
    scored_events.sort(key=lambda x: x[1], reverse=True)
    
    top_20_with_scores = scored_events[:20]

    # A 20-on felüli, de validált események összegyűjtése
    near_misses = []
    for ev, score in scored_events[20:]:
        near_misses.append(f"<b>{ev.name}</b> ({len(ev.ids)} hír) - [Rangsorolt: {score} pont]")
    
    discarded_summaries = near_misses + discarded_summaries
    
    # 6. Flash Elemzés (Mélyebb összefoglaló generálása)
    myPrint(f"🧠 Flash elemzés indítása a top {len(top_20_with_scores)} eseményre...")
    final_data_package: List[FinalEvent] = []
    
    for i, (event, score) in enumerate(top_20_with_scores, 1):
        # Az objektumból vesszük az ID-kat
        merged_ids = event.ids
        
        # Kikeressük a konkrét hír objektumokat (Article)
        relevant_news_objects = [n for n in filtered_news if n.id in merged_ids]
        
        if not relevant_news_objects:
            continue

        myPrint(f"  [{i}/{len(top_20_with_scores)}] Összefoglalás: {event.name} (Pont: {score} | Cikkek: {len(relevant_news_objects)})")
        
        # Flash modell hívása
        summary = generate_event_summary(event.name, relevant_news_objects)
        
        # Források listájának összeállítása Pydantic modellel
        sources_data = [
            ArticleSource(name=n.source, url=n.link) 
            for n in relevant_news_objects
        ]
        
        # A végleges FinalEvent objektum létrehozása
        final_data_package.append(FinalEvent(
            category=event.category,
            title=event.name,
            summary=summary,
            sources=sources_data,
            score=score
        ))

    # 7. Kimenetek (HTML generálás és publikálás)
    if final_data_package:
        myPrint(f"📦 {len(final_data_package)} esemény kész, HTML generálása...")
        
        output_handler.process_and_send(
            final_data_package=final_data_package, 
            topics_html=topics_html, 
            discarded_summaries=discarded_summaries
        )
    else:
        myPrint("⚠️ Nincs megjeleníthető adat, a HTML nem frissült.")
        
    # 8. Statisztika (Usage tracker marad a régi)
    from gemini_handler import usage_tracker        
    usage = usage_tracker.get_aggregated_stats()
    myPrint(f"📊 Token használat: {usage}")
    myPrint("✅ Kész.")

if __name__ == "__main__":
    main()