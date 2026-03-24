import config
import json
from google import genai
from google.genai import types
import time
from pydantic import BaseModel, Field
from typing import Any, List

from models import Article, EventSummaryResult, MultiClusterIdResponse, MultiClusterResponse
import llm_core
import shared_state


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
        topics = json.loads(res_text)
        if isinstance(topics, list):
            return topics
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
        schema=schema
    )
    
    try:
        if not res: return {"events": []}
        data = json.loads(res)
        # Biztosítjuk, hogy mindig legyen egy 'events' kulcsunk
        return data if "events" in data else {"events": []}
    except Exception as e:
        print(f"⚠️ JSON hiba a validációnál: {e}")
        return {"events": []}

def generate_event_summary(event_name: str, news_items: List[Article]) -> str:
    biases = []
    context_parts = []
    
    for n in news_items:
        # A config.RSS_SOURCES már tuple: (url, bias)
        source_data = config.RSS_SOURCES.get(n.source, (None, "Ismeretlen"))
        bias = source_data[1] 
        biases.append(bias)
        context_parts.append(f"FORRÁS: {n.source} ({bias})\nCÍM: {n.title}\nKIVONAT: {n.summary[:500]}\n---")

    # Dinamikus prompt meghatározása
    dynamic_instruction = get_dynamic_prompt(event_name, biases)
    
    prompt = f"""
    {dynamic_instruction}
    
    ELEMEZENDŐ ADATOK:
    {chr(10).join(context_parts)}
    """

    sys_instr = f"""
        Te egy tapasztalt, cinikus, de szigorúan objektív politikai és gazdasági elemző vagy.
        A feladatod a hírek dekonstrukciója. Ne csak azt nézd, mit írnak, hanem azt is, hogyan.
        Keresd a 'keretezési' technikákat és a politikai marketinget.
        Légy tömör és lényegretörő. Az egész elemzés ne legyen több 10-12 mondatnál.

        ELVÁRT STRUKTÚRA ÉS FORMÁTUM:
        Pár mondatban foglald össze az eseményt. Csak a közös metszetet és a megkérdőjelezhetetlen tényeket írd le.
        
        NARRATÍVÁK ÉS ELEMZÉS: 
        - Fejtsd ki a különböző politikai oldalak (konzervatív vs. liberális) tálalási módját.
        - KÜLÖNÖS FIGYELEM: Ha egy forrás a saját besorolásától eltérő (váratlanul kritikus vagy szokatlanul támogató) hangvételt üt meg, azt mindenképpen emeld ki!
        - Nevezd meg a konkrét manipulációs technikákat, érzelmi hergelést vagy elhallgatásokat.
        - Az egyes narratívák elemzése is csak 1-2 mondat legyen
        - Ha nincs érdemi különbség az oldalak között, ne gyártsd le mesterségesen, hanem írd le: 'A hír tálalása egységes'.

        ELVÁRÁSOK:
        - Nem kell bevezető ("Rendben, nézzük meg ezt az ...")
        - Kezdd az elemzést azonnal az érdemi összefoglalóval, ne írd ki fejlécként az esemény nevét (azt a rendszer automatikusan hozzáadja).
        - Ami a címben benne van azt már tudjuk, azt ne ismételd sehol.
        - Az egyes bekezdések legyenek lényegretörőek, csak pár mondat
        - Kerüld a felesleges köröket, szófordulatokat ("Fontos megjegyezni...").
        - Használj Markdown formázást (vastagítás a kulcsszavaknál).
        )
    """
        
    res = llm_core.gemini_call(
        client=client_main,
        contents=prompt,
        sys_instr=sys_instr,
        model=config.MODEL_LITE_ID,
        max_output_tokens=2048
    )
    
    return res if res else "Nem sikerült generálni az elemzést."
    
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

def translate_if_needed(text):
    """
    Lefordítja a szöveget magyarra, ha az idegen nyelvű. 
    Ha a modell üres választ ad (mert már magyar), az eredeti szöveget adja vissza.
    """
    sys_instruct = """Te egy fordító vagy. 
    FELADAT:
    1. Ha a bemeneti szöveg NEM magyar, fordítsd le magyarra.
    2. Ha a bemeneti szöveg MÁR magyar, a válaszod legyen teljesen ÜRES!
    
    SZABÁLY: Csak a fordítást küldd vissza, ne fűzz hozzá semmilyen magyarázatot vagy megjegyzést!"""
    
    res = llm_core.gemini_call(
        client=client_main,
        contents=text,
        sys_instr=sys_instruct,
        model=config.MODEL_LITE_ID,
        max_output_tokens=2048
    )
    
    # Ha kaptunk választ és nem csak üres karaktereket tartalmaz
    if res and res.strip():
        return res.strip()
    
    # Ha a válasz None vagy üres string, akkor az eredeti szöveget küldjük vissza
    return text

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
        # Itt a client_FREE-t használjuk!
        response = llm_core.gemini_call(
            client=client_free,
            contents=prompt,
            sys_instr=sys_instruct,
            model=config.MODEL_ID,
            max_output_tokens=2048,
            schema=MultiClusterIdResponse
        )
        return json.loads(response)
    except Exception as e:
        print(f"Hiba a Flash klaszterezésnél: {e}")
        return MultiClusterIdResponse(events=[])