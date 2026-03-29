import time
import random
from typing import Any, Callable, Optional
from google.genai import types
from models import SessionContext

import config

def execute_with_retry(func: Callable, *args, max_retries: int = 5, **kwargs) -> Any:
    """Univerzális retry logika."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["429", "500", "503", "quota"]) and attempt < max_retries - 1:
                wait = ((2 ** attempt) * 5) + random.uniform(0, 3)
                time.sleep(wait)
            else:
                raise e

def generate(context: SessionContext, contents: Any, sys_instr: str, schema: Any = None, model: str = config.MODEL_LITE_ID) -> Any:
    """Strukturált vagy szöveges generálás okos cache-kezeléssel."""
    
    # --- A JAVÍTÁS LÉNYEGE ---
    # Ha van aktív cache, az API nem engedélyezi a 'system_instruction' mezőt.
    # Ilyenkor a rendszerutasítást a sima prompt (contents) elejére fűzzük.
    if context.cache_id:
        final_contents = f"[RENDSZERUTASÍTÁS]\n{sys_instr}\n\n[FELADAT]\n{contents}"
        final_sys_instr = None
    else:
        final_contents = contents
        final_sys_instr = sys_instr

    gen_config = types.GenerateContentConfig(
        system_instruction=final_sys_instr, # Cache esetén ez None lesz
        cached_content=context.cache_id or None,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=schema,
        temperature=0.1,
    )
    
    # Itt már a final_contents-t küldjük be
    response = execute_with_retry(
        context.client.models.generate_content, 
        model=model, 
        contents=final_contents, 
        config=gen_config
    )
    
    context.logger.add(model, response)
    
    try:
        if schema:
            if schema != "json":
                return response.parsed
            return response.text
        return response.text
    except Exception as e:
        print(f"⚠️ Válasz feldolgozási hiba: {e}")
        return response.text if hasattr(response, 'text') else response

def embed(context: SessionContext, texts: list) -> Any:
    return execute_with_retry(context.client.models.embed_content, model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(task_type="CLUSTERING"))

def get_token_count(client, text: str, model: str = config.MODEL_LITE_ID) -> int:
    response = execute_with_retry(
        client.models.count_tokens,
        model=model,
        contents=text
    )
    return response.total_tokens

def setup_gemini_cache(client, formatted_text: str, model: str = config.MODEL_LITE_ID) -> Optional[str]:
    DISPLAY_NAME = "news_analysis_session"

    try:
        for c in client.caches.list():
            if c.display_name == DISPLAY_NAME:
                print(f"♻️ Élő cache megtalálva: {c.name}")
                return c.name
    except Exception as e:
        print(f"⚠️ Hiba a cache-ek listázásakor: {e}")

    try:
        token_count = get_token_count(client, text=formatted_text, model=model)
        print(f"📊 Bemeneti tokenek: {token_count}")
    except Exception as e:
        print(f"⚠️ Token mérés hiba: {e}")
        return None

    if token_count >= 32768:
        print(f"🧠 Küszöb felett ({token_count}): Cache létrehozása...")
        try:
            new_cache = execute_with_retry(
                client.caches.create,
                model=model,
                config={
                    "display_name": DISPLAY_NAME,
                    "contents": formatted_text, # A GenAI okos, egyből megeszi a stringet
                    "ttl": "1800s" # Fontos: Itt stringként kéri az 's'-t a másodperchez
                }
            )
            return new_cache.name
        except Exception as e:
            print(f"❌ Hiba a cache létrehozásakor: {e}")
            return None
    
    print("ℹ️ Küszöb alatt: Normál mód (nincs cache).")
    return None

def cleanup_cache(client, cache_id: str) -> None:
    if cache_id:
        try:
            client.caches.delete(name=cache_id)
            print(f"🗑️ Cache ({cache_id}) sikeresen törölve.")
        except Exception as e:
            print(f"⚠️ Hiba a cache törlésekor: {e}")