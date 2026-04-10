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

        # 1. Bemenet összeállítása (M1, M2... azonosítókkal)
        # Csak az első 3-4 hír címét küldjük el klaszterenként, hogy spóroljunk
        cluster_texts = []
        for c in clusters:
            titles = " | ".join([it.title[:100] for it in c.items[:5]])
            cluster_texts.append(f"{c.id}: [{titles}]")
        
        batch_input = "\n".join(cluster_texts)

        # 2. A "Szigorú Szerkesztő" Prompt
        sys_instr = f"""
        Feladat: Hírszerkesztő vagy. Előre csoportosított hírekről (hír-klaszterekről) kell eldöntened hogy politikai, gazdasági vagy technológiai szempontból van-e jelentőségük, vagy pedig csak zaj (bulvár, reklám, celebek, stb)
        Az egyes csoportokat egyenként vizsgáld meg, és amelyek nem minősülnek szemétnek, azoknak egy jó magyar címet kell adnod. A kimenetben elkülöníthetőnek kell lennie ID alapján hogy melyik csoportnak melyik címet adtad.
        A kimenetbe nem kell semmi bevezető, semmi magyarázat.
        
        Szabályok:
        1. Csak a VALÓDI politikai, gazdasági vagy technológiai súllyal bíró klaszterekről válaszolj.
        2. Ami bulvár, reklám, sporthír, recept vagy jelentéktelen apróság, azt HAGYD KI a válaszból.
        3. Formátum: ID: Rövid, ütős cím (max 15 szó) SZIGORÚAN MAGYAR NYELVEN!
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
        valid_map = {}
        if response and response.text:
            for line in response.text.split('\n'):
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