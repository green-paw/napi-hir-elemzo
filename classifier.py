# classifier.py
import json
from typing import List, Dict, Any
from google.genai import Client, types
import shared_state
from models import Article, SingleCluster, MultiClusterIdResponse
import config

def discover_rolling_topics(client: Client, chunk_size: int = 100) -> List[str]:
    """
    1. Fázis: Végigmegy a híreken 100-as csomagokban, és felépíti a témák listáját.
    Erős prompt-szabályokkal kényszerítjük a modellt a rövid kategórianevek használatára.
    """
    current_list: List[str] = []
    total_news: int = len(shared_state.filtered_news)

    for i in range(0, total_news, chunk_size):
        end_idx: int = min(i + chunk_size - 1, total_news - 1)
        
        # --- ÚJ, SZIGORÍTOTT PROMPT A LITE MODELLHEZ ---
        instruction: str = f"""Elemezd a híreket a(z) {i} és {end_idx} ID-k között.
            SZIGORÚ SZABÁLY: NE másold ki a hírek címét vagy szövegét! Csak rövid, 3-5 szavas, átfogó esemény-neveket (kategóriákat/témákat) generálj!
            Példa jó kimenetre: ["USA választások", "Németországi sztrájkok", "Gázai konfliktus", "Tech cégek leépítései"]"""

        if not current_list:
            prompt: str = f"{instruction}\nGyűjtsd ki az 5-10 legfontosabb egyedi eseményt."
        else:
            prompt = f"""{instruction}
            Itt az eddigi lista: {current_list}. 
            Csak teljesen ÚJ, fajsúlyos eseményeket adj hozzá. Maximum 15 elem lehet a teljes listában!"""

        # Ha nincs cache, beküldjük a nyers szöveget is
        contents: str = prompt
        if not shared_state.active_cache:
            chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
            contents = f"{prompt}\n\nHírek: {json.dumps([n.model_dump() for n in chunk], default=str)}"

        response = client.models.generate_content(
            model=config.MODEL_LITE_ID, # Használjuk a Lite modellt a config-ból
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=shared_state.active_cache.name if shared_state.active_cache else None,
                response_mime_type="application/json",
                response_schema=list[str],
                temperature=0.1, # Még alacsonyabb hőmérséklet a fegyelmezettebb válaszért
                max_output_tokens=1024 # FIZIKAI GÁT: Maximum kb. 800 szót generálhat!
            )
        )
        
        raw_text: str = response.text.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        raw_text = raw_text.strip()

        try:
            current_list = json.loads(raw_text)
            print(f"🔄 Rolling fázis: {i}-{end_idx} feldolgozva, jelenleg {len(current_list)} téma van.")
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON hiba a(z) {i}-{end_idx} blokkban: {e}")
            print(f"Nyers szöveg:\n{raw_text[:200]}... (levágva)")
            print("Folytatjuk az eddigi listával, ezt a 100-as csomagot átugorjuk.")
    
    return current_list

def refine_to_top_30(client: Client, raw_topics: List[str]) -> List[str]:
    """
    2. Fázis: A nyers témalistából kiválogatja a 30 legfontosabbat.
    """
    prompt: str = f"""
    Rangsorold és válogasd ki a maximum 30 legfontosabb eseményt ebből a listából:
    {raw_topics}
    
    Szempontok: globális hatás, rendkívüli események, társadalmi jelentőség.
    A bulvárt és a jelentéktelen ismétlődő híreket hagyd el.
    SZIGORÚ SZABÁLY: KIZÁRÓLAG a kiválasztott témák rövid neveit add vissza egy listában! Ne fűzz hozzájuk semmilyen magyarázatot vagy kommentárt!
    """
    
    response = client.models.generate_content(
        model=config.MODEL_LITE_ID, # Figyelj, hogy itt a standard modell fusson!
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=shared_state.active_cache.name if shared_state.active_cache else None,
            response_mime_type="application/json",
            response_schema=list[str],
            temperature=0.1,
            max_output_tokens=1024 # Fizikai korlát: max ~800 szó
        )
    )
    refined_list: List[str] = json.loads(response.text)
    print(f"🎯 Finomhangolás kész, {len(refined_list)} fő téma maradt.")
    return refined_list

def classify_news_with_lite(client: Client, chunk_size: int = 100) -> List[SingleCluster]:
    """
    3. Fázis: Minden hírt besorol a 30 fő téma egyikébe.
    Itt a költséghatékony gemini-2.5-flash-lite modellt használjuk!
    """
    final_clusters: List[SingleCluster] = []
    total_news: int = len(shared_state.filtered_news)

    for i in range(0, total_news, chunk_size):
        end_idx: int = min(i + chunk_size - 1, total_news - 1)
        
        prompt: str = f"""
        Rendeld hozzá a(z) {i} és {end_idx} közötti ID-val rendelkező híreket a következő témákhoz:
        {shared_state.master_topics}
        """
        
        contents: str = prompt
        if not shared_state.active_cache:
            chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
            contents = f"{prompt}\n\nHírek: {json.dumps([n.model_dump() for n in chunk], default=str)}"

        response = client.models.generate_content(
            model=config.MODEL_LITE_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                cached_content=shared_state.active_cache.name if shared_state.active_cache else None,
                response_mime_type="application/json",
                response_schema=MultiClusterIdResponse,
                temperature=0.0,
                max_output_tokens=2048 # BIZTONSÁGI GÁT: Ne tudjon végtelen ciklusba esni a JSON generálásakor
            )
        )
        
        raw_data: Dict[str, Any] = json.loads(response.text)
        
        # ID szivárgás elleni védelem (leakage protection)
        for event in raw_data.get("events", []):
            valid_ids: List[int] = [idx for idx in event["ids"] if i <= idx <= end_idx]
            if valid_ids:
                final_clusters.append(SingleCluster(title=event["title"], ids=valid_ids))
                
        print(f"🗂️ Klaszterezés fázis: {i}-{end_idx} ID-k besorolva.")
        
    return final_clusters

def clean_clusters(raw_clusters: List[SingleCluster], min_news: int = 3) -> List[SingleCluster]:
    """
    4. Fázis: Összevonja az azonos című klasztereket és kidobja a túl kicsiket.
    Tiszta Python logika, LLM hívás nélkül.
    """
    merged_dict: Dict[str, List[int]] = {}
    
    # 1. Összevonás cím alapján
    for cluster in raw_clusters:
        if cluster.title not in merged_dict:
            merged_dict[cluster.title] = []
        merged_dict[cluster.title].extend(cluster.ids)
        
    # 2. Szűrés elemszám alapján
    cleaned_list: List[SingleCluster] = []
    for title, ids in merged_dict.items():
        # Eltávolítjuk a duplikált ID-kat (biztonsági okokból)
        unique_ids: List[int] = list(set(ids))
        if len(unique_ids) >= min_news:
            cleaned_list.append(SingleCluster(title=title, ids=unique_ids))
            
    print(f"🧹 Tisztítás kész: {len(cleaned_list)} érvényes klaszter maradt (minimum {min_news} hír/klaszter).")
    return cleaned_list