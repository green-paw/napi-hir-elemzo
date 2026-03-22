# llm_core.py
from google.genai import Client, types
from typing import Any
import shared_state

def get_token_count(client: Client, model_id: str, text: str) -> int:
    """Kiszámolja a bemeneti szöveg tokenjeinek számát."""
    response = client.models.count_tokens(
        model=model_id,
        contents=text
    )
    return response.total_tokens

def setup_gemini_cache(client: Client, formatted_json_text: str, model_id: str = 'gemini-2.5-flash') -> bool:
    """
    Megméri a szöveget, és ha meghaladja a 32 768 tokent, létrehoz egy 
    Gemini Context Cache objektumot a globális memóriában.
    """
    token_count: int = get_token_count(client, model_id, formatted_json_text)
    print(f"📊 Bemeneti tokenek száma: {token_count}")

    if token_count >= 32768:
        print("🧠 Küszöb felett: Context Cache létrehozása (TTL: 1 óra)...")
        shared_state.active_cache = client.caches.create(
            model=model_id,
            config=types.CreateCacheConfig(
                display_name="napi_hir_cache",
                contents=[formatted_json_text],
                ttl="3600s"
            )
        )
        return True
    
    print("ℹ️ Küszöb alatt: Cache nem szükséges, nyers módban folytatjuk.")
    shared_state.active_cache = None
    return False

def cleanup_cache(client: Client) -> None:
    """Törli az aktív cache-t a Google szervereiről, ha létezik, hogy ne generáljon extra költséget."""
    if shared_state.active_cache is not None:
        try:
            client.caches.delete(name=shared_state.active_cache.name)
            print("🗑️ Cache sikeresen törölve.")
        except Exception as e:
            print(f"⚠️ Hiba a cache törlésekor: {e}")
        finally:
            shared_state.active_cache = None