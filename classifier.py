# classifier.py
import json
from typing import List, Dict, Any
from google.genai import Client, types
import shared_state
from models import Article, SingleCluster, MultiClusterIdResponse
import config

from llm_core import gemini_call

def discover_rolling_topics(client: Client, chunk_size: int = 100) -> List[str]:
    """
    1. Fázis: Végigmegy a híreken 100-as csomagokban, és felépíti a témák listáját.
    Erős prompt-szabályokkal kényszerítjük a modellt a rövid kategórianevek használatára.
    """
    final_list: List[str] = []
    total_news: int = len(shared_state.filtered_news)

    for i in range(0, total_news, chunk_size):
        end_idx: int = min(i + chunk_size - 1, total_news - 1)
        
        sys_instr: str = f"""
            Te egy profi független hírelemző vagy, aki a híreket csoportosítja és kategorizálja.

            FELADAT:
            - a hírek alapján eseményeket (témákat) gyűjteni, amelyek a hírek mögött állnak, összefogják az ugyanazon eseményhez kapcsolódó híreket.
            - Minden esemény legyen egy rövid mondat, max 20 szó, ami összefoglalja a mögöttes hírek lényegét. Ha országok, helyszínek, személyek szerepelnek, azokat is említsd meg a címben, de csak a legfontosabbakat!
            - Azok a hírek fontosak amelyek Magyarország gazdasági vagy politikai életére hatással vannak, vagy globális jelentőségűek (háborúk, természeti katasztrófák, gazdasági válságok, stb).
            - próbáld meg csak a legfontosabb eseményeket kigyűjteni, max 10-15 témát.

            SZIGORÚ SZABÁLY:
            NE egy az egyben a hírek címét vagy szövegét másold. Egy rövid, átfogó, max 20 szavas mondatot generálj, SZIGORÚAN MAGYAR NYELVEN!

            SZŰRÉS: bulvár, pletyka, jelentéktelen híreket NE engedj meg! Csak a legfontosabb, egyedi eseményeket gyűjtsd ki!
            """

        # Ha nincs cache, beküldjük a nyers szöveget is
        #contents: str = prompt
        #if not shared_state.active_cache:
        #    chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
        #    contents = f"{prompt}\n\nHírek: {json.dumps([n.title for n in chunk], default=str, ensure_ascii=False)}"

        chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
        news_data = [f"Cím: {n.title} | Kivonat: {n.summary[:50]}..." for n in chunk]
        contents = f"Hírek: {json.dumps(news_data, default=str, ensure_ascii=False)}"

        result_list = gemini_call(
            client=client,
            model=config.MODEL_LITE_ID,
            schema=list[str],
            sys_instr=sys_instr,
            contents=contents,
            max_output_tokens=1024
        )

        if result_list and isinstance(result_list, list):
            final_list.extend(result_list)
            print(f"✅ Hozzáadva {len(result_list)} új téma.")
        else:
            print("⚠️ Nem érkezett feldolgozható lista ebből a szakaszból.")
    
    return final_list

def refine_to_top_30(client: Client, raw_topics: list[str]) -> list[str]:
    """
    2. Fázis: Konszolidálja a redundáns témákat és kiválasztja a 30 legfontosabbat.
    """
    # Ehhez a logikai művelethez mindenképp a sima Flash kell, nem a Lite!
    model_to_use = config.MODEL_ID 

    sys_instr = """
    Te egy vezető hírszerkesztő vagy. A bemeneti listád nyers, redundáns témákat tartalmaz, 
    mivel több forrásból és több idősávból gyűjtöttük őket.
    
    FELADATOD:
    1. KONSZOLIDÁCIÓ: Ha több bejegyzés ugyanarról az eseményről szól (pl. "Forint gyengülése" és "Zuhan a magyar deviza"), vond össze őket EGYETLEN, precíz megnevezésbe.
    2. PRIORIZÁLÁS: Csak a legfontosabb (globális vagy magyar stratégiai) eseményeket tartsd meg.
    3. LIMIT: A végleges lista szigorúan maximum 30 elemű legyen.
    
    SZABÁLYOK:
    - Magyar nyelven válaszolj.
    - Egy elem max 15-20 szó legyen.
    - Töröld a bulvárt, sporthíreket és jelentéktelen helyi híreket.
    """
    
    # Kicsit strukturáltabb bemenet a modellnek
    contents = f"Íme a nyers, redundáns témalista. Vond össze az átfedéseket és add vissza a top 30-at:\n{json.dumps(raw_topics, ensure_ascii=False)}"

    result_list = gemini_call(
        client=client,
        model=model_to_use,
        schema=list[str],
        sys_instr=sys_instr,
        contents=contents,
        max_output_tokens=2048
    )
    
    if isinstance(result_list, list):
        # Biztonsági vágás, ha az LLM mégis túlszaladna
        #final_list = result_list[:30]
        print(f"🎯 Konszolidáció kész: {len(raw_topics)} nyers szál -> {len(result_list)} egyedi téma.")
        return result_list
    
    return []

def classify_news_with_lite(client: Client, chunk_size: int = 100) -> List[SingleCluster]:
    """
    3. Fázis: Minden hírt besorol a 30 fő téma egyikébe.
    Itt a költséghatékony gemini-2.5-flash-lite modellt használjuk hibatűrő logikával!
    """
    final_clusters: List[SingleCluster] = []
    total_news: int = len(shared_state.filtered_news)

    for i in range(0, total_news, chunk_size):
        end_idx: int = min(i + chunk_size - 1, total_news - 1)
        
        prompt: str = f"""
        Rendeld hozzá a(z) {i} és {end_idx} közötti ID-val rendelkező híreket a következő témákhoz, amennyiben relevánsak:
        {shared_state.master_topics}

        FONTOS: Csak akkor rendeld hozzá egy hír ID-ját egy témához, ha az esemény szorosan kapcsolódik a témához! Ne engedj meg lazán kapcsolódó besorolásokat!
        A kimenet egy JSON objektum legyen, ahol minden téma egy esemény, és az eseményekhez tartozó ID-k egy listában vannak.
        """
        
        contents: str = prompt
        if not shared_state.active_cache:
            chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
            contents = f"{prompt}\n\nHírek: {json.dumps([n.model_dump() for n in chunk], default=str)}"

        try:
            response = client.models.generate_content(
                model=config.MODEL_LITE_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    cached_content=shared_state.active_cache.name if shared_state.active_cache else None,
                    response_mime_type="application/json",
                    response_schema=MultiClusterIdResponse,
                    temperature=0.0,
                    max_output_tokens=4096 # Biztonsági gát
                )
            )
            
            # --- HIBATŰRŐ TISZTÍTÁS ÉS FELDOLGOZÁS ---
            raw_text: str = response.text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            raw_text = raw_text.strip()
            
            raw_data: Dict[str, Any] = json.loads(raw_text)
            
            # ID szivárgás elleni védelem (leakage protection)
            for event in raw_data.get("events", []):
                valid_ids: List[int] = [idx for idx in event["ids"] if i <= idx <= end_idx]
                if valid_ids:
                    final_clusters.append(SingleCluster(title=event["title"], ids=valid_ids))
                    
            print(f"🗂️ Klaszterezés fázis: {i}-{end_idx} ID-k besorolva.")
            
        except Exception as e:
            # Ha bármi hiba történik (JSON, hálózat, API vágás), jelezzük, de nem állunk le!
            print(f"⚠️ Hiba a(z) {i}-{end_idx} besorolásakor: {e}")
            print("Ezt a 100-as csomagot átugorjuk, megyünk tovább...")

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