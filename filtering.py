from pydantic import BaseModel
from typing import List
import gemini_core
from source import NewsItem

class FilterResponse(BaseModel):
    important_ids: List[str]

SYS_INSTR_FILTER = """
Te egy hírszűrő algoritmus vagy. A feladatod, hogy a megadott listából kiválogasd a VALÓDI hírértékkel bíró eseményeket.

RELEVÁNS: 
- Belföldi vagy külföldi politika, gazdaság, fontos technológiai áttörések, rendkívüli események.

NEM RELEVÁNS (Zaj):
- Bulvár, sporteredmények, időjárásjelentés, napi horoszkóp, reklámcikkek, rutinszerű tőzsdei apróhírek (pl. 0.1%-os elmozdulás).

KIMENET: 
Kizárólag egy JSON listát adj vissza a releváns hírek ID-jaival. Példa: ["C12", "C45"]
"""

def filter_lone_wolves(lone_wolves: List[NewsItem], batch_size: int = 50) -> List[NewsItem]:
    important_items = []
    
    for i in range(0, len(lone_wolves), batch_size):
        batch = lone_wolves[i : i + batch_size]
        
        # Rövidített lista készítése a promptba (ID + Cím)
        news_list_text = "\n".join([f"{item.id}: {item.title}" for item in batch])
        
        prompt = f"Válogasd ki a fontos híreket a listából:\n{news_list_text}"
        
        try:
            # Flash-lite használata a költséghatékonyság miatt
            result = gemini_core.generate(
                contents=prompt,
                sys_instr=SYS_INSTR_FILTER,
                schema=FilterResponse
            )
            
            # A visszaadott ID-k alapján kikeressük az eredeti NewsItem-eket
            valid_ids = set(result.important_ids)
            important_items.extend([item for item in batch if item.id in valid_ids])
            
        except Exception as e:
            print(f"⚠️ Hiba a magányos hírek szűrésekor (batch {i//batch_size}): {e}")
            
    return important_items