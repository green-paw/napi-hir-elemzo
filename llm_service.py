from typing import List
import re
import gemini_core
import config
from typing import Any, List, Dict, Set
from concurrent.futures import ThreadPoolExecutor
from models import BatchClassificationResponse, NewsCluster, NewsItem

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




        valid_map = {}

        for chunk in batches:
            cluster_texts = []
            current_clusters = ", ".join([c.id for c in chunk])
            for c in chunk:
                titles = "\n".join([str(it.clean_content)[:100] for it in c.items])
                cluster_texts.append(f"{c.id}: [{titles}]")
            
            batch_input = "\n".join(cluster_texts)

            # 2. A "Szigorú Szerkesztő" Prompt
            sys_instr = f"""
            Feladat: Szigorú hírszerkesztő vagy. Előre csoportosított hír-klaszterekről döntöd el, hogy van-e valódi stratégiai jelentőségük, vagy csak zajnak minősülnek.
            A cél a lényeges események szűrése és tömör magyar nyelvű összefoglalása.

            Szabályok és Prioritások:

            1. RELEVANCIA-SZŰRÉS:
            - TARTSD MEG: 
                - Magyarország bel- és külügyei (választások, pártpolitika, tüntetések, kormányzati döntések).
                - Globális konfliktusok és háborúk (Orosz-Ukrán háború, USA-Izrael-Irán konfliktus, frontvonalak, fegyverszállítások).
                - Valódi technológiai áttörések (AI, energia, űrkutatás).
                - KÖZÉLET: Súlyos balesetek, természeti katasztrófák, országos jelentőségű közbiztonsági események vagy tragédiák.
            - HAGYD KI (TRASH):
                - Külföldi politikusok egymásról alkotott magánvéleménye vagy diplomáciai szájkarate (pl. Starmer mit gondol Trumpról), ha nincs mögötte konkrét kormányzati lépés.
                - Külföldi országok lokális népszerűségi mutatói, bulvár, sport, receptek, reklámok.
                - Külföldi személyes botrányok, hacsak nem érintenek államfőt vagy nincs globális hatásuk
                - Külföldi közéleti, hacsak nem országos szintű eseményről van szó
                - Bulvár, reklám, sporthír, recept vagy jelentéktelen apróság, azt HAGYD KI a válaszból.
                - Kattintásvadász, de tartalom nélküli címek.

            2. CÍMADÁS ÉS FORMÁTUM:
            - KATEGÓRIA: POLITIKA, GAZDASÁG, TECHNOLÓGIA, KÖZÉLET vagy TRASH
            - ORSZÁG MEGJELÖLÉSE: A cím elején MINDIG szerepeljen az ország vagy régió (pl. USA:, Nagy-Britannia:, Ukrajna:). 
            - A csoport jelentősége Magyarországra vagy globális mércével egy 1-10 skálán
            - Formátum: ID: [KATEGÓRIA] [HELYSZÍN] [PONTSZÁM] Cím (Pl. M80: [POLITIKA] [Magyarország] [5]: Új közvélemény-kutatási adatok...)
            - Hossz: Max 15 szó, ütős, magyar nyelvű összefoglaló.
            - Némítás: Ha egy klaszter TRASH, semmit ne írj róla a kimenetbe (se ID-t, se magyarázatot).

            3. KIMENETI KORLÁTOK:
            - Ne írj bevezetőt, ne írj összefoglalót vagy magyarázatot a döntéseidhez.
            - Csak a valid ID-kat és a hozzájuk tartozó címeket sorold fel.
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

            print(f"{current_clusters}:\n{response}")

            # 4. Válasz feldolgozása
            if response:
                for line in response.split('\n'):
                    if ":" in line:
                        parts = line.split(":", 1)
                        mid = parts[0].strip()
                        title = parts[1].strip()
                        valid_map[mid] = title

        # 5. Státuszok beállítása az objektumokban
        for c in clusters:
            if c.id in valid_map:
                c.summary_title = valid_map[c.id]
                c.is_trash = False
            else:
                c.summary_title = ""
                c.is_trash = True

        return clusters


    def process_mini_clusters(self, clusters: List[NewsCluster]) -> List[NewsCluster]:
        """
        Batchelve küldi el a klasztereket az LLM-nek. 
        A válasz alapján frissíti a klaszterek címeit és trash státuszát.
        """
        if not clusters:
            return []        

        chunks = [clusters[i:i + 20] for i in range(0, len(clusters), 20)]
        processed: List[NewsCluster] = []

        valid_map = {}

        for chunk in chunks[:3]:
            cluster_texts = []
            for c in chunk:
                titles = " | ".join([it.title[:100] for it in c.items[:5]])
                cluster_texts.append(f"{c.id}: [{titles}]")
            
            batch_input = "\n".join(cluster_texts)

            # 2. A "Szigorú Szerkesztő" Prompt
            sys_instr = f"""
            Feladat: Szigorú hírszerkesztő vagy. Előre csoportosított hír-klaszterekről döntöd el, hogy van-e valódi stratégiai jelentőségük, vagy csak zajnak minősülnek.
            A cél a lényeges események szűrése és tömör magyar nyelvű összefoglalása.

            Szabályok és Prioritások:

            1. RELEVANCIA-SZŰRÉS:
            - TARTSD MEG: 
                - Magyarország bel- és külügyei (választások, pártpolitika, tüntetések, kormányzati döntések).
                - Globális konfliktusok és háborúk (Orosz-Ukrán háború, USA-Izrael-Irán konfliktus, frontvonalak, fegyverszállítások).
                - Valódi technológiai áttörések (AI, energia, űrkutatás).
                - KÖZÉLET: Súlyos balesetek, természeti katasztrófák, országos jelentőségű közbiztonsági események vagy tragédiák.
            - HAGYD KI (TRASH):
                - Külföldi politikusok egymásról alkotott magánvéleménye vagy diplomáciai szájkarate (pl. Starmer mit gondol Trumpról), ha nincs mögötte konkrét kormányzati lépés.
                - Külföldi országok lokális népszerűségi mutatói, bulvár, sport, receptek, reklámok.
                - Külföldi személyes botrányok, hacsak nem érintenek államfőt vagy nincs globális hatásuk
                - Külföldi közéleti, hacsak nem országos szintű eseményről van szó
                - Bulvár, reklám, sporthír, recept vagy jelentéktelen apróság, azt HAGYD KI a válaszból.
                - Kattintásvadász, de tartalom nélküli címek.

            2. CÍMADÁS ÉS FORMÁTUM:
            - KATEGÓRIA: POLITIKA, GAZDASÁG, TECHNOLÓGIA, KÖZÉLET vagy TRASH
            - ORSZÁG MEGJELÖLÉSE: A cím elején MINDIG szerepeljen az ország vagy régió (pl. USA:, Nagy-Britannia:, Ukrajna:). 
            - A csoport jelentősége Magyarországra vagy globális mércével egy 1-10 skálán
            - Formátum: ID: [KATEGÓRIA] [HELYSZÍN] [PONTSZÁM] Cím (Pl. M80: [POLITIKA] [Magyarország] [5]: Új közvélemény-kutatási adatok...)
            - Hossz: Max 15 szó, ütős, magyar nyelvű összefoglaló.
            - Némítás: Ha egy klaszter TRASH, semmit ne írj róla a kimenetbe (se ID-t, se magyarázatot).

            3. KIMENETI KORLÁTOK:
            - Ne írj bevezetőt, ne írj összefoglalót vagy magyarázatot a döntéseidhez.
            - Csak a valid ID-kat és a hozzájuk tartozó címeket sorold fel.
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
                for line in response.split('\n'):
                    if ":" in line:
                        parts = line.split(":", 1)
                        mid = parts[0].strip()
                        title = parts[1].strip()
                        valid_map[mid] = title

        # 5. Státuszok beállítása az objektumokban
        for c in clusters:
            if c.id in valid_map:
                c.summary_title = valid_map[c.id]
                c.is_trash = False
            else:
                c.summary_title = ""
                c.is_trash = True

        return clusters

    def generate_final_analysis(self, macro_clusters: List[Any]):
        """
        Itt lesz majd a 'végső nagy elemző' hívás helye.
        Ez már valószínűleg a MODEL_ID (nem lite) verziót használja majd.
        """
        # Kidolgozás alatt...
        pass