# llm_core.py
import random
import sys
import time

import config
from google.genai import Client, types
from typing import Any
import shared_state

usage_tracker = shared_state.TokenLogger()

cache_name: str = "napi_hir_cache"

def get_token_count(client: Client, model_id: str, text: str) -> int:
    """Kiszámolja a bemeneti szöveg tokenjeinek számát."""
    response = client.models.count_tokens(
        model=model_id,
        contents=text
    )
    return response.total_tokens

def setup_gemini_cache(client: Client, formatted_json_text: str, model_id: str = 'gemini-2.5-flash') -> bool:
    """
    1. Megnézi a Google szerverén, van-e már élő cache-ünk.
    2. Ha nincs, megméri a szöveget, és ha > 32k, létrehoz egy újat.
    """
    # 1. Lekérdezzük a szerverről az élő cache-eket
    try:
        for existing_cache in client.caches.list():
            if existing_cache.display_name == cache_name:
                print(f"♻️ Élő cache megtalálva a Google szerverén! Újracsatlakozás: {existing_cache.name}")
                shared_state.active_cache = existing_cache
                return True
    except Exception as e:
        print(f"⚠️ Hiba a cache-ek lekérdezésekor: {e}")

    # 2. Ha nem találtunk (vagy lejárt), jöhet a mérés és létrehozás
    token_count: int = get_token_count(client, model_id, formatted_json_text)
    print(f"📊 Bemeneti tokenek száma: {token_count}")

    if token_count >= 32768:
        print("🧠 Küszöb felett: Context Cache létrehozása (TTL: 1 óra)...")
        shared_state.active_cache = client.caches.create(
            model=model_id,
            config=types.CreateCachedContentConfig(
                display_name=cache_name,
                contents=[formatted_json_text],
                ttl="3600s"
            )
        )
        return True
    
    print("ℹ️ Küszöb alatt: Cache nem szükséges, nyers módban folytatjuk.")
    shared_state.active_cache = None
    return False

def cleanup_cache(client: Client) -> None:
    """Törli az aktív cache-t a Google szervereiről."""
    if shared_state.active_cache is not None:
        try:
            client.caches.delete(name=shared_state.active_cache.name)
            print("🗑️ Cache sikeresen törölve a Google szervereiről.")
        except Exception as e:
            print(f"⚠️ Hiba a cache törlésekor: {e}")
        finally:
            shared_state.active_cache = None

def gemini_call(client: Client, model: str = config.MODEL_LITE_ID, schema: Any = None, sys_instr: str = "Te egy objektív, független hírelemző vagy.", contents: str = "", max_output_tokens: int = 1024) -> Any:
    if not contents:
        print("⚠️ Figyelmeztetés: Nincs bemeneti tartalom a Gemini híváshoz.")
        return ""

    response = None
    max_retries: int = 5
    attempt = 0

    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH")
        for cat in [
            "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_HARASSMENT", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
            "HARM_CATEGORY_DANGEROUS_CONTENT", 
            "HARM_CATEGORY_CIVIC_INTEGRITY"
        ]
    ]

    config_args = {
                "system_instruction": sys_instr,
                "response_mime_type": "application/json" if schema else "text/plain",
                "temperature": 0.1 if schema else 0.3,
                "max_output_tokens": max_output_tokens,
                "safety_settings": safety_settings,
            }
            
    if schema:
        config_args["response_schema"] = schema
        
    # A büntetést CSAK a normál flash modellnél alkalmazzuk, a lite-nál nem!
    #if "lite" not in model.lower():
    #    config_args["frequency_penalty"] = 1.0

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_args)
            )
            break  # Ha sikeres, kilépünk a retry loopból
        except Exception as e:
            if "429" in str(e):
                wait = ((2 ** attempt) * 10) + random.uniform(0, 5)
                print(f"🚫 Kvóta elfogyott (429). Hosszabb pihenő: {wait:.1f} mp... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "503" in str(e) or "500" in str(e):
                wait = (attempt + 1) * 5
                print(f"⚠️ Szerver hiba (50x). Újrapróbálkozás {wait} mp múlva... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                # Minden más végzetes hiba (pl. jogosultság, rossz paraméter)
                print(f"❌ Végzetes hiba történt: {e}")
                sys.exit(1)

    if response is None and attempt == max_retries - 1:
        print("❌ Kritikus: Minden újrapróbálkozás sikertelen volt.")
        sys.exit(1)

    try:
        if response is not None and hasattr(response, 'usage_metadata'):
            usage_tracker.add(model, response)
            print(f"📊 {model} | Output tokens: {response.usage_metadata.candidates_token_count} | Cached input: {getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0}")
        if response is not None and hasattr(response, 'candidates') and len(response.candidates) > 0:
            if response.candidates[0].finish_reason == "MAX_TOKENS":
                print(f"⚠️ FIGYELMEZTETÉS ({model}): A válasz túl hosszú, le lett vágva!")            
    except:
        pass

    # 3. OKOS VISSZATÉRÉS
    if response is not None:
        if schema:
            try:
                # Ha van schema, a .parsed adja a tiszta Python típust (pl. listát)
                return response.parsed
            except Exception as e:
                print(f"⚠️ Parsing hiba, fallback nyers szövegre: {e}")
                return response.text.strip()
        
        return response.text.strip()
    else:
        print("⚠️ Visszatérés előtt: Nincs érvényes válasz a Gemini-tól.")
        return ""
