from typing import TypedDict, List
from google import genai
from clustering import MacroCluster
import gemini_core
from google.genai import types
from source import NewsItem

class AnalysisResult(TypedDict):
    reconstruction: str
    narrative_games: str
    manipulation_log: str
    objectivity_score: int

def analyze_macro_cluster(macro: MacroCluster) -> AnalysisResult:
    """
    Vak elemzést végez egy MacroCluster tartalmán.
    A bemenet a makróban lévő összes egyedi NewsItem tartalma.
    """
    client = genai.Client(api_key="YOUR_API_KEY")

    # Az összes NewsItem kigyűjtése a mikrókból (flattening)
    # A mikrókban lévő 90%-os hasonlóság miatt itt érdemes lehet 
    # mikrónként csak az első 1-2 hírt bevenni, ha túl sok a token.
    all_news_items: List[NewsItem] = [
        item for micro in macro.micro_clusters for item in micro
    ]

    # Szöveges kontextus építése: Cím + Tartalom + ID
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

    response = gemini_core.generate(sys_instr=system_instruction, contents=full_context, max_output_tokens=2048, schema=AnalysisResult)
    return response
