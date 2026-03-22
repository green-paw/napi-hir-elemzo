# summarizer.py
import json
import time
from typing import List, Dict, Any
from google.genai import Client, types
import shared_state
from models import Article, SingleCluster, Summary
import config

def generate_final_reports(client: Client, valid_clusters: List[SingleCluster]) -> List[Summary]:
    """
    5. Fázis: Kettesével haladva (batching) legenerálja a végső összefoglalókat
    a Free Tier gemini-2.5-flash modellel, figyelve a 15 RPM limitre.
    """
    all_summaries: List[Summary] = []
    total_clusters: int = len(valid_clusters)

    print(f"✍️ Összesen {total_clusters} klaszter vár összefoglalásra.")

    for i in range(0, total_clusters, 2):
        batch: List[SingleCluster] = valid_clusters[i : i + 2]
        
        # Előkészítjük a promptot a batch-hez
        prompt_data: str = ""
        for idx, cluster in enumerate(batch):
            # Kikeressük a konkrét hír objektumokat az ID-k alapján a globális memóriából
            cluster_news: List[Article] = [
                news for news in shared_state.filtered_news if news.id in cluster.ids
            ]
            prompt_data += f"\n--- {idx+1}. ESEMÉNY: {cluster.title} ---\n"
            prompt_data += json.dumps([n.model_dump() for n in cluster_news], default=str)

        prompt: str = f"""
        Kaptál híreket az alábbi esemény(ek)hez. 
        Írj mindegyikhez egy-egy profi, max 2500 karakteres, lényegre törő összefoglalót!
        Használj bullet pointokat a legfontosabb mérföldkövekhez. 
        A kimenet mindenképp tartalmazza a forrásként használt cikkek ID-jait is.
        
        Adatok:
        {prompt_data}
        """

        try:
            response = client.models.generate_content(
                model=config.MODEL_LITE_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[Summary], 
                    temperature=0.3,
                    max_output_tokens=4096 # BIZTONSÁGI GÁT: 2 összefoglaló sosem lehet ennél hosszabb!
                )
            )
            
            # A JSON válasz feldolgozása Pydantic objektumokká
            raw_summaries: List[Dict[str, Any]] = json.loads(response.text)
            for item in raw_summaries:
                all_summaries.append(Summary(**item))
                
            print(f"✅ Kész: {[c.title for c in batch]}")

            # Rate limit védelem (15 RPM -> max 15 hívás / perc)
            # A 4 másodperces alvás garantálja, hogy bőven a limit alatt maradunk.
            if i + 2 < total_clusters:
                print("⏳ Várakozás a Rate Limit miatt (4 másodperc)...")
                time.sleep(4)

        except Exception as e:
            print(f"❌ Hiba az összefoglalásnál ({[c.title for c in batch]}): {e}")

    return all_summaries