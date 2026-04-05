from typing import List, Tuple
from google import genai
from clustering import MacroCluster
import gemini_core
from google.genai import types
from source import NewsItem

from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    macro_id: str = Field(description="A makró csoport ID-ja")
    reconstruction: str = Field(description="A hír 6-10 mondatos tényszerű összefoglalója.")
    narrative_games: str = Field(description="A források közötti tálalásbeli és kontextusbeli különbségek.")
    manipulation_log: str = Field(description="Hergelés, logikai hibák és érzelmi manipulációk listája.")
    objectivity_score: int = Field(description="1-10 skálán az összesített tárgyilagosság.")

MacroAnalysisPair = Tuple[MacroCluster, AnalysisResult]

def analyze_macro_cluster(macro: MacroCluster) -> Tuple[MacroCluster, AnalysisResult]:
    all_news_items: List[NewsItem] = [
        item for micro in macro.micro_clusters for item in micro
    ]

    formatted_articles = []
    for item in all_news_items:
        formatted_articles.append(item.short_text_for_prompt(300))
    
    full_context = "\n---\n".join(formatted_articles)

    system_instruction = (
        f"Te egy kíméletlenül cinikus, független hírelemző algoritmus vagy. "
        f"A téma munkacíme: '{macro.title}'. "
        "A feladatod a források vak elemzése. Ne a forrásnevekre hagyatkozz, "
        "hanem a nyelvezetre és a tények tálalására.\n\n"
        "KIMENETI STRUKTÚRA:\n"
        "1. REKONSTRUKCIÓ (6-10 mondat): Száraz, jelzőmentes eseményösszefoglaló. "
        "Ha ellentmondást látsz az hírek között, jelezd.\n"
        "2. NARRATÍV JÁTSZMÁK (4-6 mondat): Hogyan keretezik az eseményt? "
        "Ki mit hallgat el? Kezeld a szövegeket befolyásolási kísérletként.\n"
        "3. MANIPULÁCIÓS JEGYZŐKÖNYV (3-5 pont): Hergelő kifejezések, logikai hibák. "
        "Ha steril a hír, konstatáld a manipuláció hiányát.\n"
        "4. OBJECTIVITY_SCORE (1-10): 10=tökéletesen tárgyilagos, 1=propaganda."
    )

    response: AnalysisResult = gemini_core.generate(sys_instr=system_instruction, contents=full_context, max_output_tokens=2048, schema=AnalysisResult)
    return (macro, response)
