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

    def process_clusters_with_llm(self, clusters: List[NewsCluster]) -> List[NewsCluster]:
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
            - HAGYD KI (TRASH): 
                - Külföldi politikusok egymásról alkotott magánvéleménye vagy diplomáciai szájkarate (pl. Starmer mit gondol Trumpról), ha nincs mögötte konkrét kormányzati lépés.
                - Külföldi országok lokális népszerűségi mutatói, bulvár, sport, receptek, reklámok.
                - Kattintásvadász, de tartalom nélküli címek.

            2. CÍMADÁS ÉS FORMÁTUM:
            - ORSZÁG MEGJELÖLÉSE: A cím elején MINDIG szerepeljen az ország vagy régió (pl. USA:, Nagy-Britannia:, Ukrajna:). 
            - Formátum: ID: [KATEGÓRIA] Cím (Pl. M80: [POLITIKA] [Magyarország]: Új közvélemény-kutatási adatok...)
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