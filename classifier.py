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
            Te egy magyar stratégiai elemző vagy. A feladatod a globális és hazai hírek szűrése a magyar döntéshozók számára.

            FELADAT:
            - a hírek alapján eseményeket (témákat) gyűjteni, amelyek a hírek mögött állnak, összefogják az ugyanazon eseményhez kapcsolódó híreket.
            - Minden esemény legyen egy rövid mondat, max 20 szó, ami összefoglalja a mögöttes hírek lényegét. Ha országok, helyszínek, személyek szerepelnek, azokat is említsd meg a címben, de csak a legfontosabbakat!
            - Azok a hírek fontosak amelyek Magyarország gazdasági vagy politikai életére hatással vannak, vagy globális jelentőségűek (háborúk, természeti katasztrófák, gazdasági válságok, stb).
            - próbáld meg csak a legfontosabb eseményeket kigyűjteni, max 10-15 témát.

            SZIGORÚ SZŰRÉSI KRITÉRIUMOK:
            1. MAGYAR RELEVANCIA: Minden hazai politikai, gazdasági hír jöhet.
            2. GLOBÁLIS HATÁS: Csak olyan külföldi hír maradhat, ami:
            - Közvetlen hatással van az olaj/gázárakra vagy a forintra.
            - Világhatalmi átrendeződést mutat (USA, Kína, Oroszország, EU magállamok).
            - Szomszédos országok (pl. Ukrajna) háborús helyzete.
            3. TÖRLENDŐ (Irreleváns): 
            - Más országok belügyei (pl. szlovén/francia/brit helyi választások, adózási szabályok, helyi bűnügyek).
            - Lokális balesetek (pl. LaGuardia reptér, londoni tűzeset).
            - Bulvár, tech-kütyük, egyedi céges hírek (kivéve ha piaci összeomlást okoznak).
            
            SZIGORÚ SZABÁLY:
            NE egy az egyben a hírek címét vagy szövegét másold. Egy rövid, átfogó, max 20 szavas mondatot generálj, SZIGORÚAN MAGYAR NYELVEN!
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
            max_output_tokens=2048
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

    sys_instr = """
    Te egy magyar stratégiai elemző vagy. A feladatod a globális és hazai hírek szűrése a magyar döntéshozók számára.
    A bemeneti listád nyers, redundáns témákat tartalmaz, mivel több forrásból és több idősávból gyűjtöttük őket.

    FELADATOD: Vond össze az átfedéseket, és csak a 30 legfontosabb, stratégiailag releváns pontot tartsd meg!
    1. KONSZOLIDÁCIÓ: Ha több bejegyzés ugyanarról az eseményről szól (pl. "Forint gyengülése" és "Zuhan a magyar deviza"), vond össze őket EGYETLEN, precíz megnevezésbe.
    2. PRIORIZÁLÁS: Csak a legfontosabb (globális vagy magyar stratégiai) eseményeket tartsd meg.
    3. LIMIT: A végleges lista szigorúan maximum 30 elemű legyen.

    SZIGORÚ SZŰRÉSI KRITÉRIUMOK:
    1. MAGYAR RELEVANCIA: Minden hazai politikai, gazdasági hír jöhet.
    2. GLOBÁLIS HATÁS: Csak olyan külföldi hír maradhat, ami:
    - Közvetlen hatással van az olaj/gázárakra vagy a forintra.
    - Világhatalmi átrendeződést mutat (USA, Kína, Oroszország, EU magállamok).
    - Szomszédos országok (pl. Ukrajna) háborús helyzete.
    3. TÖRLENDŐ (Irreleváns): 
    - Más országok belügyei (pl. szlovén/francia/brit helyi választások, adózási szabályok, helyi bűnügyek).
    - Lokális balesetek (pl. LaGuardia reptér, londoni tűzeset).
    - Bulvár, tech-kütyük, egyedi céges hírek (kivéve ha piaci összeomlást okoznak).

    SZABÁLYOK:
    - Magyar nyelven válaszolj.
    - Egy elem max 15-20 szó legyen.
    - Töröld a bulvárt, sporthíreket és jelentéktelen helyi híreket.
    """
    
    # Kicsit strukturáltabb bemenet a modellnek
    contents = f"Íme a nyers, redundáns témalista. Vond össze az átfedéseket és add vissza a top 30-at:\n{json.dumps(raw_topics, ensure_ascii=False)}"

    result_list = gemini_call(
        client=client,
        model=config.MODEL_LITE_ID,
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
    Használja a gemini_call-t és a MultiClusterIdResponse sémát.
    """
    final_clusters: List[SingleCluster] = []
    total_news: int = len(shared_state.filtered_news)

    # Szigorú rendszerutasítás a besoroláshoz
    sys_instr: str = f"""
    Te egy precíz hírszerkesztő vagy. A feladatod a hírek (ID és tartalom) besorolása a megadott témák alá.

    TÉMÁK (MASTER LIST):
    {shared_state.master_topics}

    SZIGORÚ SZABÁLYOK:
    1. Csak a megadott témák közül választhatsz! Ne találj ki új témát.
    2. Ha egy hír nem illik tökéletesen egyik témához sem, hagyd ki (NE sorold be sehova)!
    3. Különösen figyelj a relevanciára: a helyi baleseteket, bulvárt és irreleváns külföldi híreket (pl. brit/amerikai helyi ügyek) dobd el, hacsak nem illeszkednek szorosan egy globális/stratégiai témához.
    4. Egy hír ID-ja csak egyetlen témához tartozhat.
    
    FORMÁTUM: A megadott JSON sémát használd (events lista, benne a title és az ids lista).
    """

    for i in range(0, total_news, chunk_size):
        end_idx: int = min(i + chunk_size - 1, total_news - 1)
        
        # Csak a címet és a summary elejét küldjük, hogy spóroljunk a tokennel
        chunk: List[Article] = shared_state.filtered_news[i : end_idx + 1]
        news_payload = [
            {"id": n.id, "title": n.title, "text": (n.summary or "")[:100]} 
            for n in chunk
        ]

        contents = f"Sorold be az alábbi híreket az ID-k alapján:\n{json.dumps(news_payload, ensure_ascii=False)}"

        # Hívás a közös függvényeddel
        # A MultiClusterIdResponse sémát használjuk a modellek.py-ból
        response_data: MultiClusterIdResponse = gemini_call(
            client=client,
            model=config.MODEL_LITE_ID,
            schema=MultiClusterIdResponse,
            sys_instr=sys_instr,
            contents=contents,
            max_output_tokens=4096
        )

        # Ellenőrizzük, kaptunk-e adatot (MultiClusterIdResponse objektumot várunk)
        if response_data and hasattr(response_data, 'events'):
            count_in_batch = 0
            for event in response_data.events:
                # ID szivárgás elleni védelem: csak az ebben a batchben lévő ID-kat fogadjuk el
                valid_ids: List[int] = [idx for idx in event.ids if i <= idx <= end_idx]
                if valid_ids:
                    final_clusters.append(SingleCluster(title=event.title, ids=valid_ids))
                    count_in_batch += len(valid_ids)
            
            print(f"🗂️ Klaszterezés: {i}-{end_idx} tartomány kész. ({count_in_batch} hír besorolva)")
        else:
            print(f"⚠️ Hiba vagy üres válasz a(z) {i}-{end_idx} tartományban.")

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