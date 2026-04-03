from clustering import MacroCluster
import gemini_core

from pydantic import BaseModel, Field
from typing import Dict, List

from source import NewsItem

def generate_macro_label(macro: MacroCluster) -> str:

    sys_instr = """
    Te egy precíz hír-elemző és rendszerező modul vagy. A feladatod, hogy egy hírcsoportból (Makró klaszter) egyetlen, tömör és beszédes magyar nyelvű gyűjtőcímet generálj, ne legyen hosszabb egy mondatnál.
    Valamint adj egy Impact Score-t (1-10) a csoportnak, Magyarország vagy globális hatás szempontjából.

    Pontozási szempontok:
    10: Világháborús veszély, világformáló technológia (Artemis, AGI), magyar államcsőd/kormányváltás esélye.
    7-9: Jelentős háborús eszkaláció, globális gazdasági válság jelei, nagy magyar politikai botrányok.
    4-6: Fontos, de hétköznapibb hírek (választási kampány eseményei, tőzsdei mozgások, nagyobb céges hírek).
    1-3: Technikai adatközlések, rutin pénzügyi jelentések, lokális (nem magyar) balesetek, töltelék hírek.

    Szabályok:
    Nyelvfüggetlenség: Bármilyen nyelvű híreket kapsz, a kimenet mindig magyar legyen.
    Standardizálás: Kerüld a sallangokat (pl. "Hírek a...", "Beszámoló erről:"). Használj tárgyilagos, újságírói stílust.
    Összevonhatóság: Törekedj arra, hogy ha a hírek egy globális eseményről szólnak (pl. Artemis-program vagy Iráni konfliktus), a cím legyen alkalmas arra, hogy más, hasonló témájú csoportokkal is egybeessen.
    Specifikusság: Ha a csoport egy konkrét eseményről szól (pl. "Trump kirúgta Pam Bondit"), ne csak annyit írj, hogy "Amerikai politika".

    Kimenet: PONT | CÍM
    (példa: 8 | Trump totális vámháborút hirdetett az EU ellen)
    Csak egy sor, mindenféle magyarázat vagy formázás nélkül.
    """

    all_text: list[str] = []
    for micro in macro.micro_clusters:
        for item in micro:
            all_text.append(f"{item.title} - {item.content[:100]}")

    prompt = f"Generálj egy közös magyar címet és adj pontszámot ezeknek a híreknek:\n" + "\n".join(all_text)
    
    label = gemini_core.generate(
        contents=prompt,
        sys_instr=sys_instr,
        max_output_tokens=256
    )

    if not label or not label.strip():
        representative_micro = max(macro.micro_clusters, key=len)
        if representative_micro and len(representative_micro) > 0:
            label = representative_micro[0].title
            macro.embedding = representative_micro[0].embedding
        else:
            label = "Vegyes hírek"

    macro.title = label.strip()
    return macro.title














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