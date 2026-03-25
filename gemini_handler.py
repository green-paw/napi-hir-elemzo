import config
import json
from google import genai
from google.genai import types
import time
from typing import Any, Dict, List

from models import Article, MultiClusterIdResponse, MultiClusterResponse
from models import StructuredEventSummary
import llm_core


# 1. Globális kliens létrehozása itt, a handlerben
client_main = genai.Client(api_key=config.GOOGLE_API_KEY)
client_free = genai.Client(api_key=config.GOOGLE_API_KEY_FREE)

def get_strategic_topics(titles_sample):
    sys_instr = f"""
    Te egy stratégiai politikai, gazdasági elemző vagy.
    Elemezd a következő hírcímeket, és határozz meg maximum 7 darab kiemelt stratégiai fókuszpontot, amelyek a mai napot dominálják.

    Szigorú prioritási sorrend:
    - Geopolitika és Háború: Katonai konfliktusok, eszkaláció, nemzetközi szankciók, Irán, Ukrajna, USA-Kína feszültség.
    - Magyar Stratégiai Érdek: Hazai belpolitikai válságok, választási kampány, kormányzati döntések, nemzetbiztonság.
    - Kritikus Gazdaság és Energia: Infláció, forint-összeomlás, energiabiztonság, olajárak, nagyvállalati (OTP, MOL, CATL) krízishelyzetek.
    
    Tiltólista:
    - Csak akkor említs technológiát, klímát vagy kultúrát, ha az közvetlen, súlyos gazdasági vagy politikai következménnyel jár (pl. AI-szabályozás miatti tőzsdei bukás).
    - Ha nincs 7 valóban stratégiai téma, adj kevesebbet, de ne töltsd fel bulvárral vagy irreleváns "trendekkel".
    
    VÁLASZ FORMÁTUMA:
    Csak a témák címeit add vissza, fontossági sorrendben, JSON listában.
    Példa: ["Téma 1", "Téma 2", "Téma 3"]
    """

    prompt: str = f"""
    HÍREK:
    {titles_sample}
    """
    
    res_text = llm_core.gemini_call(
        client=client_main,
        model=config.MODEL_LITE_ID,
        contents=prompt, 
        sys_instr=sys_instr,
        schema=list[str]
    )

    try:
        if isinstance(res_text, list):
            return res_text
        return []
    except Exception as e:
        print(f"❌ Hiba a témák feldolgozásánál: {e}")
        print(res_text)
        return []

def validate_news_clusters(cluster_data: str, schema=MultiClusterResponse) -> dict:
    """Lite modell: Szétválasztja a matematikai klasztert valódi eseményekre és pontoz."""
    
    sys_instruct = """Te egy cinikus, de tűpontos hírszerkesztő algoritmus vagy. 
    A feladatod, hogy egy matematikai módszerrel összegyűjtött hírkupacból (nyers klaszter) kihámozd a VALÓDI, különálló eseményeket.

    STRATÉGIAI SZABÁLYOK:
    1. SZÉTVÁLASZTÁS: Ha a hírek között több különböző vállalat (pl. BMW vs. CATL), különálló incidens vagy téma van, KÖTELESSÉGED őket külön eseményként (event) visszaadni a listában. Ne gyárts "Debreceni ipar" típusú gyűjtőneveket!
    2. RELEVANCIA SZŰRÉS: Csak azokat az eseményeket tartsd meg, amiknek a Relevance pontszáma legalább 4. A bulvárt, baleseteket, jelentéktelen színes híreket egyszerűen hagyd ki (ne adj nekik eseményt).
    3. TISZTÍTÁS: Ha egy hír (ID) nem illik szorosan egyik eseményhez sem, ne kényszerítsd bele sehová, hagyd el.

    PONTOZÁSI ÚTMUTATÓ (1-10):
    - RELEVANCE: 10 = kritikus magyar érdek/világpolitika. 1 = bulvár, celeb, egyéni sors.
    - IMPACT (1-10): 
        Az impact pontszám meghatározásakor legyél kíméletlen:
        9-10: Háborús cselekmény, több száz halott, globális gazdasági összeomlás, atomfenyegetés.
        7-8: Államfők bejelentései, Magyarország és más országok közötti politikai/gazdasági jelentős események. Országos politikai földindulás, kritikus infrastruktúra leállása (pl. Kuba áramszünet), deviza-összeomlás.
        4-6: Vállalati eredmények, helyi szabályozások, sportesemények (pl. BL döntő), környezeti hírek (pl. fafajok).
        1-3: Bulvár, érdekességek, technológiai apróságok.
    - NOVELTY: Mennyire friss és tényalapú az információ?

    VÁLASZ: Kizárólag a megadott JSON sémát használd!"""

    # Fontos: itt már a MultiClusterResponse sémát használjuk!
    res = llm_core.gemini_call(
        client=client_main,
        contents=cluster_data,
        sys_instr=sys_instruct,
        model=config.MODEL_LITE_ID,
        schema=schema,
        max_output_tokens=1500
    )
    
    try:
        if not res: 
            return {"events": []}
        return res.model_dump()
    except Exception as e:
        print(f"⚠️ JSON hiba a validációnál: {e}")
        return {"events": []}
    
def get_gemini_embeddings(texts):
    """Vektorok lekérése újrapróbálkozási logikával."""
    all_embeddings = []
    
    # 100-as batch-ek (ez jó)
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        
        # Belső újrapróbálkozás a 429-es hiba kezelésére
        for attempt in range(5):
            try:
                response = client_main.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="CLUSTERING")
                )
                all_embeddings.extend([embedding.values for embedding in response.embeddings])
                
                # Siker esetén várunk egy kicsit, hogy ne fussunk bele a következő limitbe
                time.sleep(1) 
                break # Kilépünk az attempt ciklusból
                
            except Exception as e:
                if "429" in str(e) or "exhausted" in str(e).lower():
                    wait_time = (attempt + 1) * 10 # 10, 20, 30... mp várakozás
                    print(f"⚠️ Embedding kvóta elfogyott, várakozás {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Kritikus hiba az embedding során: {e}")
                    raise e
                    
    return all_embeddings

def get_dynamic_prompt(event_name, source_biases):
    # Meghatározzuk, milyen típusú forrásaink vannak
    has_right = any("konzervatív" in b or "kormánypárti" in b for b in source_biases)
    has_left = any("liberális" in b or "baloldali" in b or "kritikai" in b for b in source_biases)
    
    base_info = f"Esemény: {event_name}\n"

    # A: ÜTKÖZTETŐ PROMPT (Ha mindkét oldal jelen van)
    if has_right and has_left:
        return base_info + """
        KÜLDETÉS: NARRATÍVA-ÜTKÖZTETÉS. 
        Mivel a források között markáns politikai különbség van, a feladatod:
        1. Emeld ki a két oldal közötti értelmezési különbséget.
        2. Keresd meg a manipulációt: ki hergel, ki hallgat el tényeket?
        3. Szűrd le a tiszta tényeket, amiben mindenki egyetért.
        """
    
    # B: "BUBORÉK" PROMPT (Ha csak az egyik oldal ír róla)
    elif has_right or has_left:
        side = "jobboldali" if has_right else "baloldali"
        return base_info + f"""
        KÜLDETÉS: KRITIKAI ELLENSÚLY. 
        Erről az eseményről jelenleg csak {side} források számoltak be a klaszterben. 
        1. Emiatt legyél fokozottan gyanakvó: mi lehet a "vakfolt"? 
        2. Milyen érdekeket szolgálhat ez a tálalás? 
        3. Próbáld meg azonosítani a túlzásokat.
        """
    
    # C: ÁLTALÁNOS / NEMZETKÖZI PROMPT
    else:
        return base_info + """
        KÜLDETÉS: OBJEKTÍV ÖSSZEGZÉS. 
        Elemezd a híreket tényalapú megközelítéssel, fókuszálj a geopolitikai és gazdasági hatásokra.
        """

def batch_cluster_news(formatted_news: str) -> MultiClusterIdResponse:
    """Ez a függvény végzi a nagy, egyben történő klaszterezést az INGYENES kulccsal."""
    sys_instruct = f"""
    Te egy precíz hírszerkesztő vagy. Csak a megadott sémát használd.

    Itt a mai hírek listája azonosítókkal.
    
    FELADATOK:
    1. SZŰRÉS: Szigorúan hagyd figyelmen kívül a bulvár, sport, divat, időjárás, életmód és egyéb érdektelen témákat. Ezeket a híreket dobd el!
    2. KLASZTEREZÉS: A megmaradt, komoly híreket csoportosítsd KÜLÖNÁLLÓ ESEMÉNYEK szerint.
    
    SZABÁLYOK:
    - Csak azokat a híreket tedd egy csoportba, amik PONTOSAN ugyanarról a konkrét eseményről szólnak.
    - Ami nem illik sehova (zaj), azt hagyd ki. Ne csinálj "Egyéb" jellegű gyűjtő kategóriákat!
    """

    prompt = f"""
    HÍREK:
    {formatted_news}
    """
    
    try:
        response = llm_core.gemini_call(
            client=client_main,
            contents=prompt,
            sys_instr=sys_instruct,
            model=config.MODEL_LITE_ID,
            max_output_tokens=2048,
            schema=MultiClusterIdResponse
        )
        return json.loads(response)
    except Exception as e:
        print(f"Hiba a Flash klaszterezésnél: {e}")
        return MultiClusterIdResponse(events=[])

def generate_structured_summary(event_name: str, news_items: List[Article]) -> Dict[str, Any]:
    biases: List[str] = []
    context_parts: List[str] = []
    
    # Változók a token optimalizáláshoz
    has_left: bool = False
    has_right: bool = False
    
    for n in news_items:
        source_data: tuple = config.RSS_SOURCES.get(n.source, (None, "Ismeretlen"))
        bias: str = source_data[1] 
        biases.append(bias)
        
        # Gyors ellenőrzés: Milyen típusú forrásaink vannak?
        bias_lower: str = bias.lower()
        if "bal" in bias_lower or "liberális" in bias_lower or "kritikai" in bias_lower:
            has_left = True
        if "jobb" in bias_lower or "konzervatív" in bias_lower or "kormánypárti" in bias_lower:
            has_right = True
            
        context_parts.append(f"FORRÁS: {n.source} ({bias})\nCÍM: {n.title}\nKIVONAT: {n.summary[:500]}\n---")

    # A dinamikus promptod az eseményhez
    #dynamic_instruction: str = get_dynamic_prompt(event_name, biases)
    
    # A tokenkímélő utasítások most már a user promptba kerülnek, így a system prompt fix marad!
    left_instruction: str = (
        "Elemezd a baloldali/liberális narratívát a 'left_wing_analysis' mezőben, emeld ki a manipulációs technikákat." 
        if has_left else 
        "NINCS baloldali forrás a listában. A 'left_wing_analysis' mező értéke SZIGORÚAN csak egy üres string ('') legyen, ne találj ki semmit!"
    )
    
    right_instruction: str = (
        "Elemezd a jobboldali/konzervatív narratívát a 'right_wing_analysis' mezőben, emeld ki a fókuszt és az érveket." 
        if has_right else 
        "NINCS jobboldali forrás a listában. A 'right_wing_analysis' mező értéke SZIGORÚAN csak egy üres string ('') legyen, ne találj ki semmit!"
    )

    prompt: str = f"""
    ELEMZÉSI SZABÁLYOK AZ ADOTT ESEMÉNYHEZ:
    - {left_instruction}
    - {right_instruction}
    
    ELEMEZENDŐ ADATOK:
    {chr(10).join(context_parts)}
    """

    # Ez itt egy 100%-ig fix, statikus szöveg, tökéletes a Gemini Caching számára!
    sys_instr: str = """
        Te egy tapasztalt, cinikus, de szigorúan objektív politikai és gazdasági elemző vagy.
        A feladatod a hírek dekonstrukciója és a tények leválasztása a narratívákról.

        ELVÁRÁSOK:
        - Nem kell bevezető ("Rendben, nézzük meg ezt az ...")
        - Kezdd az elemzést azonnal az érdemi összefoglalóval, ne írd ki fejlécként az esemény nevét (azt a rendszer automatikusan hozzáadja).
        - Ami a címben benne van azt már tudjuk, azt ne ismételd sehol.
        - Az egyes bekezdések legyenek lényegretörőek, csak pár mondat.
        - Kerüld a felesleges köröket, szófordulatokat ("Fontos megjegyezni...").
        - Használj Markdown formázást (vastagítás a kulcsszavaknál).
        
        SZABÁLYOK A VÁLASZHOZ (JSON SÉMA):
        1. 'summary': SZIGORÚAN CSAK A TÉNYEK (max 500 karakter). Nincs vélemény, nincs forráselemzés. Próbáld megtalálni amiben minden forrás egyetért, vagy ha különböznek a vélemények, akkor a közös metszetet. Ne ismételd meg a címből már ismert információkat!
        2. 'left_wing_analysis' és 'right_wing_analysis':
            - Kövesd a promptban kapott specifikus utasításokat!
            - Fejtsd ki a különböző politikai oldalak (konzervatív vs. liberális) tálalási módját.
            - KÜLÖNÖS FIGYELEM: Ha egy forrás a saját besorolásától eltérő (váratlanul kritikus vagy szokatlanul támogató) hangvételt üt meg, azt mindenképpen emeld ki!
            - Nevezd meg a konkrét manipulációs technikákat, érzelmi hergelést vagy elhallgatásokat.
            - Az egyes narratívák elemzése is csak pár mondat legyen
            - Ha nincs érdemi különbség az oldalak között, ne gyártsd le mesterségesen, hanem írd le: 'A hír tálalása egységes'.
    """
        
    res: Any = llm_core.gemini_call(
        client=client_main,
        contents=prompt,
        sys_instr=sys_instr,
        model=config.MODEL_LITE_ID,
        schema=StructuredEventSummary,
        max_output_tokens=2048
    )
    
    """
    # Ha a Free kliens elakadt (pl. politikai szűrő miatt levágta a JSON-t)
    if not res or isinstance(res, str):
        print("⚠️ A Free kliens elhasalt (valószínűleg politikai szűrő). Próba a Main (fizetős) kulccsal...")
        
        # 2. Próba a fizetős klienssel
        res = llm_core.gemini_call(
            client=client_main,
            contents=prompt,
            sys_instr=sys_instr,
            model=config.MODEL_LITE_ID,   # Itt a fizetős Lite-ot használjuk költséghatékonyságból
            schema=StructuredEventSummary,
            max_output_tokens=2048
        )
    """

    if not res or isinstance(res, str): 
        raise ValueError("Mindkét kliens üres vagy hibás választ adott.")

    try:
        return res.model_dump()
    except Exception as e:
        print(f"⚠️ Hiba a summary generálásánál: {e}")
        return {
            "title": event_name, 
            "summary": "Hiba történt az elemzés során.", 
            "left_wing_analysis": "", 
            "right_wing_analysis": "", 
            "category": "EGYÉB", 
            "score": 0
        }