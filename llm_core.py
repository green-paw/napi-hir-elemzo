# llm_core.py
from google.genai import Client, types
from typing import Any
import shared_state

cache_name: str = "napi_hir_cache_lite"

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