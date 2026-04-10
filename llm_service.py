import re
import gemini_core
import config
from typing import Any, List, Dict, Set
from concurrent.futures import ThreadPoolExecutor
from models import BatchClassificationResponse, NewsItem

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

    def classify_news_batch(self, items: List[NewsItem], trash_bin: Dict[str, Set[str]]) -> List[NewsItem]:
        """A beérkező hírek gyors, kötegelt osztályozása."""
        if not items:
            return []

        valid_items = []
        id_map = {f"C{i}": item for i, item in enumerate(items)}
        item_ids = list(id_map.keys())
        chunks = [item_ids[i:i + 30] for i in range(0, len(item_ids), 30)]

        first_run_logged = False

        def _process_chunk(chunk_ids: List[str]):
            nonlocal first_run_logged
            news_block = "\n".join([f"{bid}: {id_map[bid].title}" for bid in chunk_ids])
            
            # Sémát használunk a regex helyett
            parsed_response = gemini_core.generate(
                contents=f"Osztályozd a következő híreket:\n{news_block}",
                sys_instr=PROMPTS["classifier"],
                schema=BatchClassificationResponse, # Itt adjuk át a Pydantic modellt
                model=config.MODEL_LITE_ID
            )
            
            if not first_run_logged and parsed_response:
                print("\n--- [DEBUG] Első LLM Válasz (Séma szerint) ---")
                # A Pydantic modelleket a .model_dump_json() segítségével szépen ki lehet íratni
                print(parsed_response.model_dump_json(indent=2))
                print("-------------------------------------------\n")
                first_run_logged = True

            # parsed_response már egy BatchClassificationResponse objektum
            for res in parsed_response.results:
                if res.id in id_map:
                    item = id_map[res.id]
                    
                    # 1. Értékadás a rövidített séma alapján (res.cat és res.hun)
                    item.category = res.cat 
                    item.profile["is_hun"] = 1.0 if res.hun == "HUN" else 0.0
                    item.profile["is_checked"] = 1.0
                    
                    # 2. Szűrés - a sémában megadott 'cat' mezőt vizsgáljuk
                    if res.cat == "TRASH":
                        trash_bin.setdefault("TRASH", set()).add(item.hash)
                    else:
                        # Csak a releváns hírek kerülnek a következő feldolgozási szakaszba
                        valid_items.append(item)

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(_process_chunk, chunks)

        return valid_items

    def generate_final_analysis(self, macro_clusters: List[Any]):
        """
        Itt lesz majd a 'végső nagy elemző' hívás helye.
        Ez már valószínűleg a MODEL_ID (nem lite) verziót használja majd.
        """
        # Kidolgozás alatt...
        pass