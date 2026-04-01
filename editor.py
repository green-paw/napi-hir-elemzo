import gemini_core

from pydantic import BaseModel, Field
from typing import Dict, List

from source import NewsItem

class RefinedEvent(BaseModel):
    summary: str = Field(description="Az esemény rövid, tényszerű összefoglalója magyarul (1 mondat).")
    micro_cluster_indices: List[int] = Field(description="A mikro-klaszterek sorszámai (0-tól indulva), amik ebbe az eseménybe tartoznak.")
    importance_score: int = Field(description="Az esemény hírértéke 1-10 között (gazdasági/politikai súly).")

class ClusterValidationResponse(BaseModel):
    is_split_needed: bool = Field(description="True, ha a bemeneti csoport több különálló eseményt tartalmaz.")
    events: List[RefinedEvent] = Field(description="Az események listája. Ha nem kell bontani, akkor is egy elem szerepel itt.")

SYS_INSTR_EDITOR = """
Te egy tapasztalt hírszerkesztő vagy. A feladatod a kapott hír-csoportok (makro-klaszterek) validálása.
Egy csoport több 'mikro-klasztert' tartalmaz. Minden mikro-klaszter azonos tartalmú hírek gyűjteménye.

SZABÁLYOK:
1. ELEMZÉS: Olvasd el a mikro-klaszterek reprezentatív szövegeit.
2. DÖNTÉS: Ha a mikro-klaszterek valóban ugyanarról a konkrét eseményről szólnak, tartsd őket egyben.
3. SZÉTBONTÁS: Ha a matematikai algoritmus hibásan mosott össze két különböző eseményt (pl. két külön törvényjavaslat vagy két eltérő ország hírei), bontsd őket külön 'RefinedEvent' objektumokba.
4. ÖSSZEFOGLALÓ: Minden 'RefinedEvent'-hez írj egy pontos, száraz, újságírói stílusú összefoglalót magyarul.
5. RELEVANCIA: Csak a politikai, gazdasági vagy jelentős társadalmi híreket tartsd meg.
"""


def validate_and_refine_clusters(macro_clusters: List[List[List[NewsItem]]]) -> List[Dict]:
    final_events = []

    for macro_idx, macro_cluster in enumerate(macro_clusters):
        # Prompt összeállítása: csak a mikro-klaszterek első hírét küldjük el
        cluster_info = ""
        for i, micro in enumerate(macro_cluster):
            rep = micro[0] # Reprezentáns hír
            sources = ", ".join([m.source_id for m in micro])
            cluster_info += f"\n--- MIKRO-KLASZTER {i} ---\n"
            cluster_info += f"Források: {sources}\n"
            cluster_info += f"Szöveg: {rep.title}. {rep.content[:400]}...\n"

        user_prompt = f"Validáld a következő hír-csoportot:{cluster_info}"

        try:
            # A gemini_core.generate hívása a Pydantic sémával
            result: ClusterValidationResponse = gemini_core.generate(
                contents=user_prompt,
                sys_instr=SYS_INSTR_EDITOR,
                schema=ClusterValidationResponse
            )

            # Az eredmények feldolgozása
            for event_data in result.events:
                # Visszakeressük a híreket az indexek alapján
                involved_news = []
                for m_idx in event_data.micro_cluster_indices:
                    if m_idx < len(macro_cluster):
                        involved_news.extend(macro_cluster[m_idx])
                
                final_events.append({
                    "summary": event_data.summary,
                    "importance": event_data.importance_score,
                    "news_items": involved_news
                })

        except Exception as e:
            print(f"❌ Hiba a(z) {macro_idx}. makro-klaszter validálásakor: {e}")

    return final_events