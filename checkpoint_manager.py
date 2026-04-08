from datetime import datetime
import os
import json
import argparse
from typing import Any, Type, TypeVar, Optional, Generic
from pydantic import TypeAdapter, BaseModel

# CLI paraméterek beolvasása (pl. python main.py --force)
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Ignorálja a meglévő cache fájlokat és újat generál')
args, unknown = parser.parse_known_args()

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Ez egy generikus tároló, ami bármilyen 'T' típust elnyel
class CacheWrapper(BaseModel, Generic[T]):
    timestamp: datetime
    data: T

def load_checkpoint(filename: str, expected_type: Any) -> Optional[Any]:
    # 1. Beolvassuk a környezeti változókat
    current_branch = os.getenv("CURRENT_BRANCH", "main")
    settings_str = os.getenv("CACHE_SETTINGS", "{}")

    print(f"CACHE SETTINGS: {settings_str}")
    
    use_cache_env = True # Alapértelmezett érték, ha valami hiányzik
    
    # 2. JSON feldolgozása
    try:
        settings = json.loads(settings_str)
        val = settings.get(current_branch, True)
        use_cache_env = str(val).lower() == "true"
    except json.JSONDecodeError:
        print("⚠️ Hiba a CACHE_SETTINGS JSON formátumában. Alapértelmezett Cache = True.")

    # 3. Döntés a betöltésről
    if not use_cache_env or args.force:
        print(f"⚠️ [CACHE KIKAPCSOLVA: {current_branch} ágon] {filename} betöltése átugorva.")
        return None
        
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Cache betöltve: {filepath}")
                
                # A TypeAdapter simán megeszi a List[Article] típust is
                adapter = TypeAdapter(expected_type)
                return adapter.validate_python(data)
        except Exception as e:
            print(f"⚠️ Hiba a cache betöltésekor ({filepath}): {e}")
    
    return None

def save_checkpoint(filename: str, data: Any, expected_type: Any = None) -> None:
    """Elmenti a Pydantic modellt vagy listát JSON formátumban."""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    # Ha explicit megadjuk a típust (pl. List[Article]), azt használja, különben kitalálja
    adapter = TypeAdapter(expected_type if expected_type else type(data))
    json_bytes = adapter.dump_json(data, indent=4)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_bytes.decode('utf-8'))
    print(f"💾 Cache mentve: {filepath}")
