import os
import json
import argparse
from typing import Any, Type, TypeVar, Optional
from pydantic import TypeAdapter, BaseModel

T = TypeVar('T', bound=BaseModel)

# CLI paraméterek beolvasása (pl. python main.py --force)
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Ignorálja a meglévő cache fájlokat és újat generál')
args, unknown = parser.parse_known_args()

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_checkpoint(filename: str, expected_type: Any) -> Optional[Any]:
    """Betölti a megadott JSON-t és a kért Pydantic/Python típusra alakítja."""
    if args.force:
        print(f"⚠️ [--force] Aktív: {filename} betöltése átugorva.")
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