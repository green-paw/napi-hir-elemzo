from typing import List
import re
import gemini_core
import config
from typing import Any, List, Dict, Set
from concurrent.futures import ThreadPoolExecutor
from models import BatchClassificationResponse, LLMClusterResponse, NewsCluster, NewsItem
import json

# Itt gyűjtheted a különböző feladatokhoz tartozó promptokat
PROMPTS = {
    "classifier": """
        Osztályozd a híreket ID alapján. 
        Kategóriák: POL, ECO, TEC, TRASH.
        Lokáció: HUN, INT.
        Válaszformátum: ID:KATEGÓRIA:LOKÁCIÓ
    """,
    "summarizer": """
        Készíts vezetői összefoglalót a megadott makró-klaszter eseményeiből...
    """,
    "auditor": """
        Ellenőrizd a hírcsoport konzisztenciáját...
    """
}

class LLMService:
    """
    Az üzleti logika és az LLM közötti híd. 
    A gemini_core réteget használja az alacsony szintű műveletekhez.
    """
    def process_large_clusters(self, clusters: List[NewsCluster], limit: int = 30) -> List[NewsCluster]:
        if not clusters:
            return []
        
        sorted_clusters: List[NewsCluster] = sorted(clusters, key=lambda c: len(c.items))
        
        batches: List[List[NewsCluster]] = []
        current_batch: List[NewsCluster] = []
        current_count: int = 0
        half_limit: float = limit / 2

        for cluster in sorted_clusters:
            size: int = len(cluster.items)

            # Ha a klaszter kisebb, mint a limit fele, próbáljuk tömöríteni
            if size < half_limit:
                if current_count + size <= limit:
                    current_batch.append(cluster)
                    current_count += size
                else:
                    # Ha nem fér be a kicsi a mostani kicsik mellé, lezárjuk
                    batches.append(current_batch)
                    current_batch = [cluster]
                    current_count = size
            else:
                # Ha a klaszter >= 15, akkor a jelenlegi gyűjtőt (ha van) lezárjuk,
                # mert ez a nagy klaszter mellé már valószínűleg nem férne be más
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_count = 0
                
                # A nagy klaszter megy a saját batch-ébe
                batches.append([cluster])

        # Maradék kicsik mentése
        if current_batch:
            batches.append(current_batch)

        if not batches: return clusters

        processed: List[NewsCluster] = []

        for chunk in batches:
            cluster_texts = []
            for c in chunk:
                short_text = ",\n".join([it.json_for_clustering() for it in c.items])
                cluster_texts.append(f"{c.id}: [{short_text}]")
            
            batch_input = "\n".join(cluster_texts)

            # 2. A "Szigorú Szerkesztő" Prompt
            sys_instr = f"""
            Feladat: Szigorú hírszerkesztő vagy. Előre csoportosított hír-klaszterekről döntöd el, hogy van-e valódi stratégiai politikai vagy gazdasági jelentőségük, vagy csak zajnak minősülnek.
            A cél a lényeges események szűrése és tömör magyar nyelvű összefoglalása.

            Szabályok és Prioritások:

            1. KLASZTEREK KEZELÉSE ÉS SZÉTBONTÁSA:
                A klaszterek embedding alapján kerültek csoportosításra, amiben benne van a hibalehetőség. Ha egy klaszterben több, egymástól független stratégiai hír van (pl. két külön politikai esemény), bontsd szét őket!
                Szétbontás esetén az eredeti ID-t egészítsd ki (pl. M12 -> M12_a, M12_b).
                A kimeneti JSON-ben tüntesd fel az adott al-csoporthoz tartozó hírek belső ID-it (pl. item_ids: [C1, C5]).
            
            2. SÚLYOZÁSI MECHANIZMUS (Értékelési mátrix)
                Minden klasztert pontozz az alábbi két szempont szerint, majd ezek együttes súlyát mérlegeld. Ha az össz pontszám 5 vagy kevesebb akkor a klaszter TRASH-nek tekinthető.

                A) Hatókör (Scope): Kit és mekkora területet érint?
                    1-3 pont (Lokális/Egyéni): Magánszemélyek, celebek, egyedi bűncselekmények, barlangi mentés, közlekedési balesetek, helyi (városi szintű) problémák.
                    Példa: "Machetés támadás NYC-ben", "Tom Cruise Balatonfüreden".
                    4-6 pont (Regionális/Szakmai): Egy egész szektort érintő hírek, nagyvárosi szintű beruházások, szakmai viták, amik nem borítják fel az ország életét.
                    Példa: "Változik a parkolási rend Budapesten", "Új funkciót kap az OpenAI".
                    7-10 pont (Országos/Globális): Államszintű döntések, háborús frontvonalak, választási eredmények, nemzetközi szerződések, globális gazdasági trendek.
                    Példa: "Magyar választási szabályok módosítása", "Irán válaszcsapást jelentett be".

                B) Hatás (Impact): Milyen típusú és mélységű a változás, Magyarország vagy globális szempontból:
                    1-3 pont (Zaj/Adat): Bulvár, sport, életmód, időjárás, tőzsdei napi ingadozás, szoftverfrissítések, receptek.
                    Példa: "A tőzsdei árfolyamok ma stagnáltak", "Tippmix tanácsok hétvégére".
                    4-6 pont (Társadalmi/Közérzeti): Nagyobb tömegeket érintő, de nem politikai változások, közérdekű közlekedési fennakadások, nagyobb sztrájkok, jelentős de nem stratégiai technológiai hírek.
                    Példa: "Leállt a vasúti forgalom a Dunántúlon", "Újabb tüntetést szerveznek a tanárok".
                    7-10 pont (Stratégiai/Egzisztenciális): Sorsfordító események: kormányváltás esélye, háborús eszkaláció, országos adótörvények, rendszerszintű gazdasági válság vagy fellendülés.
                    Példa: "Orosz áttörés Donyeckben", "A jegybank váratlanul kamatot emelt".

            2. SZŰRÉSI SZABÁLYOK:
                TARTSD MEG
                    Magyarországi vagy nagyhatalmi stratégiai politika, gazdaság, választások.
                    Háborús konfliktusok frontvonalbeli változásai.
                    Valódi technológiai vagy tudományos áttörések.
                TRASH (Azonnal vesd el, ha bármelyik teljesül):
                    Lokális esemény: Egyedi bűnügyek (pl. gyilkosság, késelés), balesetek, tragédiák, barlangi mentések, amiknek nincs országos politikai hatása.
                    Alacsony hatás: Bulvár, celeb-hírek, sport, divat, horoszkóp, receptek, tőzsdei adatsorok, időjárás.
                    Üres retorika: Diplomáciai "szájkarate", puszta vélemények vagy sárdobálás konkrét kormányzati lépés nélkül.
                    Technikai: RSS hibák, tartalom nélküli kattintásvadász címek.

            3. KIMENETI FORMÁTUM (JSONL):
                Kizárólag nyers JSONL formátumban válaszolj. Minden sor egy JSON objektum legyen az alábbi mezőkkel:
                    id: A klaszter (vagy al-klaszter) azonosítója (string).
                    score: Az általad kalkulált súlyozott pontszám (int, 1-10).
                    title: a cím elején az érintett ország/régió, majd egy ütős összefoglaló (max 15 szó), SZIGORÚAN magyar nyelven! (Pl. [USA] Trump bejelentette a harmadik világháborút)
                    item_ids: Az eredeti hírek ID-jai, amik ebbe a (szétbontott) klaszterbe tartoznak (list of strings). Ha a klaszter nem bontott, ez maradjon üresen.
                Kimeneti korlátok: Ne írj bevezetőt, ne használj markdown kódblokkokat (```), csak a tiszta JSONL sorokat sorold fel.
            """

            sys_instr = """
            Te egy stratégiai, politikai és gazdasági hírszerkesztő vagy. A feladatod vegyes hírekből különálló, logikailag összeillő csoportokat alkotni.

            Lépések:
            1. próbálj néhány kulcsszót, maximum 5-öt kiemelni a hírekból (ki? hol? mit csinált?), amelyek alapján több hír egy konzisztens eseménnyé áll össze.
            2. a bejövő adatok ebben a szerkezetben érkeznek: csoport ID: [egyes hírek id-ja, címe és a tartalom eleje].
            3. Kizárólag nyers JSONL formátumban válaszolj. Minden sor egy JSON objektum legyen az alábbi mezőkkel:
                id: A klaszter új azonosítója (string), a csoport eredeti azonosítója után betűjellel megkülönböztetve (pl: be: M12 -> ki: M12_a, M12_b, stb)
                title: a kulcsszavak amiket azonosítottál (string, pl: "Trump, USA, Katonai fejlesztések bejelentése"), SZIGORÚAN MAGYAR NYELVEN
                item_ids: Az eredeti hírek ID-jai, amik ebbe az új klaszterbe tartoznak (list of strings).
            4. Kimeneti korlátok: Ne írj bevezetőt, ne használj markdown kódblokkokat (```), csak a tiszta JSONL sorokat sorold fel.

            Példa a kimenetre:
            {"id": "M12_a", "title":"Orbán, Budapest, Beszédett mondott", "item_ids": [C1, C5, C12]}
            """

            prompt = f"""
            Hírek:
            {batch_input}
            """

            # 3. Gemini hívás (Flash Lite ideális erre)
            response = gemini_core.generate(
                sys_instr=sys_instr,
                contents=prompt,
                max_output_tokens=2048
            )

            print(response)

            # 4. Válasz feldolgozása
            if response:
                results = process_llm_output(response)
                processed.extend(merge_llm_responses(chunk, results))

        return processed

    def generate_final_analysis(self, macro_clusters: List[Any]):
        """
        Itt lesz majd a 'végső nagy elemző' hívás helye.
        Ez már valószínűleg a MODEL_ID (nem lite) verziót használja majd.
        """
        # Kidolgozás alatt...
        pass

def process_llm_output(raw_text: str) -> List[LLMClusterResponse]:
    # Markdown kódblokkok és üres sorok eltávolítása
    clean_lines = [
        line.strip() 
        for line in raw_text.replace("```jsonl", "").replace("```", "").splitlines() 
        if line.strip()
    ]
    
    parsed_objects: List[LLMClusterResponse] = []
    for line in clean_lines:
        try:
            # Pydantic V2 parse_raw helyett model_validate_json
            obj = LLMClusterResponse.model_validate_json(line)
            parsed_objects.append(obj)
        except Exception as e:
            print(f"Hiba a sor feldolgozásakor: {e} | Sor: {line}")
            
    return parsed_objects

def merge_llm_responses(
    original_clusters: List[NewsCluster], 
    llm_responses: List[LLMClusterResponse]
) -> List[NewsCluster]:
    
    # 1. Keresőtérkép építése: original_id -> NewsCluster
    cluster_lookup = {c.id: c for c in original_clusters}
    
    # 2. Hír keresőtérkép építése: item_id -> NewsItem (az összes klaszterből)
    item_lookup = {}
    for c in original_clusters:
        for item in c.items:
            item_lookup[item.id] = item

    final_clusters = []

    for resp in llm_responses:
        # Az eredeti ID kinyerése (pl. "M12_a" -> "M12")
        base_id = resp.id.split('_')[0]
        original_cluster = cluster_lookup.get(base_id)
        
        if not original_cluster:
            continue

        # Meghatározzuk, mely hírek tartoznak ebbe az (al)klaszterbe
        if resp.item_ids:
            # Csak azokat a híreket válogatjuk be, amiket az LLM felsorolt
            subset_items = [item_lookup[iid] for iid in resp.item_ids if iid in item_lookup]
        else:
            # Ha nincs item_ids, az eredeti klaszter összes híre marad
            subset_items = original_cluster.items

        # Új NewsCluster példányosítása a válasz alapján
        new_cluster = NewsCluster(cluster_id=resp.id, items=subset_items)
        
        # LLM metaadatok átvezetése
        new_cluster.summary_title = resp.title if resp.title else f"[{resp.reason}]"
        new_cluster.is_trash = resp.score < 7 # A te súlyozási logikád alapján
        
        final_clusters.append(new_cluster)

    return final_clusters