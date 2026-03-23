import json
import time
from typing import List
from google.genai import Client
from llm_core import gemini_call
import shared_state
from models import Article, SingleCluster, Summary
import config

def generate_final_reports(client: Client, valid_clusters: List[SingleCluster]) -> List[Summary]:
    """
    5. Fázis: Legenerálja a végső, publikálható összefoglalókat.
    Kettesével halad, hogy optimalizálja a hívásokat.
    """
    all_summaries: List[Summary] = []
    total_clusters: int = len(valid_clusters)

    print(f"✍️ Összesen {total_clusters} klaszter vár összefoglalásra.")

    sys_instr = """
    Te egy profi online újságszerkesztő vagy. A feladatod, hogy a megadott hírcsoportokból 
    tömör, lényegre törő, objektív összefoglalót írj MAGYAR nyelven.
    
    STRUKTÚRA:
    - Cím: Erős, informatív cím (maradhat az eredeti, vagy javítsd fel).
    - Szöveg: 2-4 mondat, ami összefoglalja a történések lényegét. 
    - Használj bullet pointokat, ha több különböző esemény van egy témán belül.
    - A végén ne legyen semmilyen üdvözlés vagy extra duma.
    
    FORMÁTUM: Használd a megadott Summary sémát (listában várjuk az objektumokat).
    """

    for i in range(0, total_clusters, 2):
        batch: List[SingleCluster] = valid_clusters[i : i + 2]
        
        # Adatok előkészítése a modellnek
        prompt_data = []
        for cluster in batch:
            cluster_news = [n for n in shared_state.filtered_news if n.id in cluster.ids]
            news_texts = [f"[{n.source}]: {n.title} - {n.summary[:200]}" for n in cluster_news]
            prompt_data.append({
                "original_title": cluster.title,
                "news_count": len(cluster.ids),
                "content": news_texts,
                "ids": cluster.ids
            })

        contents = f"Készíts összefoglalót az alábbi {len(batch)} eseményről:\n{json.dumps(prompt_data, ensure_ascii=False)}"

        # Hívás a közös függvényeddel (schema=list[Summary] használatával)
        result: List[Summary] = gemini_call(
            client=client,
            model=config.MODEL_ID, # Sima Flash az összefoglaláshoz
            schema=list[Summary],
            sys_instr=sys_instr,
            contents=contents,
            max_output_tokens=4096
        )

        if result and isinstance(result, list):
            all_summaries.extend(result)
            print(f"✅ Kész: {[c.title[:40] + '...' for c in batch]}")
        else:
            print(f"⚠️ Sikertelen összefoglaló a batchnél: i={i}")

        # Rövid pihenő a Rate Limit (RPM) miatt (ingyenes Tier esetén fontos)
        time.sleep(2)

    return all_summaries