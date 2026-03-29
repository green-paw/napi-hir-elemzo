import models
import gemini_core
import gemini_services
import config
from google.genai import Client

import models # Namespace tudatos

def create_session(news_list: list, api_key: str) -> models.SessionContext:
    """
    A teljes inicializálási folyamat: Kliens létrehozása, 
    vektorizálás és Context Cache setup.
    """
    # 1. Alapok létrehozása
    client = Client(api_key=api_key)
    logger = models.TokenLogger()
    
    # Dict-é alakítjuk az ID-k alapján a gyors eléréshez
    articles_dict = {a.id: a for a in news_list}
    
    # Létrehozzuk az alap kontextust (egyelőre cache nélkül)
    context = models.SessionContext(
        articles=articles_dict,
        cache_id="", # Ezt mindjárt kitöltjük
        client=client,
        token_logger=logger
    )
    
    # 2. VEKTORIZÁLÁS (Embedding)
    # A gemini_services modulon keresztül lekérjük az összes hír vektorát
    print(f"💠 {len(news_list)} hír vektorizálása folyamatban...")
    gemini_services.fill_embeddings(context)
    
    # 3. CONTEXT CACHE SETUP
    # Formázzuk a szöveget a Gemini számára
    formatted_text = _prepare_cache_text(articles_dict)
    
    # A gemini_core segítségével létrehozzuk vagy lekérjük a cache-t
    # Ez a függvény már tartalmazza a 32k token ellenőrzést és a retry-t
    cache_id = gemini_core.setup_gemini_cache(
        client=client, 
        formatted_text=formatted_text,
        model=config.MODEL_LITE_ID
    )
    
    # Frissítjük a kontextust a cache azonosítóval (ha sikerült)
    context.cache_id = cache_id or ""
    
    return context

def _prepare_cache_text(articles_dict: dict) -> str:
    """Belső segédfüggvény a hírek szöveges formázásához a cache számára."""
    header = "DATABASE OF NEWS ARTICLES (USE THESE FOR ANALYSIS):\n"
    separator = "-" * 20 + "\n"
    
    lines = [header]
    for a in articles_dict.values():
        lines.append(f"ID: {a.id}")
        lines.append(f"SOURCE: {a.source}")
        lines.append(f"TITLE: {a.title}")
        lines.append(f"CONTENT: {a.summary}")
        lines.append(separator)
        
    return "\n".join(lines)