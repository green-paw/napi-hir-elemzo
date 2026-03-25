import os
import json
import argparse
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

# CLI paraméterek beolvasása (pl. python main.py --force)
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Ignorálja a meglévő cache fájlokat és újat generál')
args, unknown = parser.parse_known_args()

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_checkpoint(filename: str, model_class: Type[T]) -> Optional[T]:
    """Betölti a megadott JSON-t és Pydantic modellé alakítja."""
    if args.force:
        print(f"⚠️ [--force] Aktív: {filename} betöltése átugorva.")
        return None
        
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Cache betöltve: {filepath}")
                return model_class.model_validate(data)
        except Exception as e:
            print(f"⚠️ Hiba a cache betöltésekor ({filepath}): {e}")
    
    return None

def save_checkpoint(filename: str, data: BaseModel) -> None:
    """Elmenti a Pydantic modellt JSON formátumban."""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(data.model_dump_json(indent=4))
    print(f"💾 Cache mentve: {filepath}")