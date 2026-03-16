import json
import config

from google import genai
from google.genai import types
import time
from pydantic import BaseModel, Field
from typing import List

class Scores(BaseModel):
    relevance: int = Field(description="Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából (1-10)")
    impact: int = Field(description="Az esemény súlya (1-10)")
    novelate: int = Field(description="Mennyire tartalmaz új információt (1-10)")

class ClusterResult(BaseModel):
    name: str = Field(description="Esemény neve és helyszíne")
    category: str = Field(description="Kategória: HAZAI, GLOBÁLIS vagy EGYÉB")
    scores: Scores
    ids: List[int] = Field(description="A csoportba ténylegesen beleillő hírek ID-jai")

class TokenLogger:
    def __init__(self):
        self.log = []

    def add(self, model_name, response):
        usage = response.usage_metadata
        self.log.append({
            "model": model_name,
            "input": usage.prompt_token_count,
            "output": getattr(usage, 'candidates_token_count', 0) # Embeddingnél nincs output
        })

    def get_summary(self):
        return self.log

# Hozz létre egy példányt a modul szintjén
usage_tracker = TokenLogger()

# 1. Globális kliens létrehozása itt, a handlerben
client = genai.Client(
    api_key=config.GOOGLE_API_KEY, 
    http_options={'api_version': 'v1beta'}
)

def _gemini_engine(prompt, sys_instruct, model_type="lite", is_json=False, schema=None):
    model_name = "gemini-2.5-flash-lite" if model_type == "lite" else "gemini-2.5-flash"

    # Újrapróbálkozási logika (maximum 5 kísérlet)
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    max_output_tokens=800,
                    temperature=0.0 if is_json else 0.2,
                    response_mime_type="application/json" if is_json else "text/plain",
                    response_schema=schema if is_json and schema else None
                )
            )
            
            usage_tracker.add(model_name, response)            
            return response.text

        except Exception as e:
            error_msg = str(e).lower()
            
            # Csak ezeknél a hibáknál érdemes újrapróbálkozni (Network/Rate limit)
            retry_errors = ["503", "429", "quota", "overloaded", "unavailable"]
            
            if any(err in error_msg for err in retry_errors):
                wait_time = (attempt + 1) * 5
                print(f"⚠️ Szerver túlterhelt, várakozás {wait_time}s... ({attempt+1}/5)")
                time.sleep(wait_time)
            else:
                # KRITIKUS HIBA: Azonnali leállás (pl. NameError, SyntaxError, Auth error)
                print(f"❌ KRITIKUS HIBA ({model_name}): {e}")
                # Kényszerített leállás, hogy ne pörögjön a ciklus
                raise SystemExit(1) 
           
    return None

def get_strategic_topics(titles_sample):
    prompt = f"""
    Az alábbi hírcímek alapján azonosítsd a 7 legfontosabb stratégiai, politikai vagy gazdasági témát.
    Elsősorban Magyarország politikai és gazdasági érintettsége a fontos, valamint a globális konfliktusok és jelentős gazdasági események.
    Ilyen vagy hasonló témák, mint "Technológiai fejlődés", "Társadalmi és kulturális trendek" akkor legyenek benne ha tényleg nincs jobb.
    Ezek nem kellenek: Bulvár, pletykák, click-bait.
    
    HÍREK:
    {titles_sample}
    
    VÁLASZ FORMÁTUMA:
    Egy JSON listát adj vissza, ami csak a témák nevét tartalmazza, semmi mást!
    Példa: ["Téma 1", "Téma 2", "Téma 3"]
    """
    
    # Használd az is_json=True paramétert, amit már beépítettünk a _gemini_engine-be!
    res_text = _gemini_engine(prompt, "Te egy stratégiai, politikai vagy gazdasági elemző vagy.", is_json=True)
    
    try:
        import json
        topics = json.loads(res_text)
        if isinstance(topics, list):
            return topics
        return []
    except Exception as e:
        print(f"❌ Hiba a témák feldolgozásánál: {e}")
        return []

def validate_news_clusters(cluster_data, schema):
    """Lite modell: Stratégiai szempontok alapján pontozza a klasztereket."""
    
    sys_instruct = """Te egy tapasztalt hírszerkesztő vagy. 
    A feladatod a hírcsoportok validálása és szigorú pontozása gazdasági és politikai szempontból.

    PONTOZÁSI ÚTMUTATÓ:
    1. RELEVANCE (1-10): 
       - 10: Kritikus magyar gazdasági/politikai esemény, globális háborús eszkaláció.
       - 1: Személyes történetek, bulvár, egyéni sorsok, érdekességek (pl. esküvő, celeb hír).
       - HA A HÍR CSAK EGYÉNI SZINTŰ (hiába háborús övezet), NEM KAPHAT 4-NÉL MAGASABB PONTOT!

    2. IMPACT (1-10): 
       - 10: Milliókat érintő döntés, országos jelentőség.
       - 1: Csak az érintett személyekre vagy egy szűk körre van hatása.

    3. NOVELTY (1-10): Mennyire hoz friss, eddig nem ismert tényeket.

    SZABÁLYOK:
    - Ha a hír bulvár jellegű vagy emberi érdekesség (human interest), büntesd alacsony pontszámokkal minden kategóriában!
    - A 'name' mező értéke MINDEN ESETBEN magyar nyelvű legyen, akkor is, ha a források vagy a téma nemzetközi!
    """

    # Itt hívjuk meg a motort
    res = _gemini_engine(cluster_data, sys_instruct, model_type="lite", is_json=True, schema=schema)
    
    try:
        return json.loads(res) if res else {}
    except Exception as e:
        print(f"⚠️ JSON hiba: {e}")
        return {}

def generate_event_summary(event_name, news_items):
    biases = []
    context_parts = []
    
    for n in news_items:
        # A config.RSS_SOURCES már tuple: (url, bias)
        source_data = config.RSS_SOURCES.get(n['source'], (None, "Ismeretlen"))
        bias = source_data[1] 
        biases.append(bias)
        context_parts.append(f"FORRÁS: {n['source']} ({bias})\nCÍM: {n['title']}\nKIVONAT: {n['summary'][:500]}\n---")

    # Dinamikus prompt meghatározása
    dynamic_instruction = get_dynamic_prompt(event_name, biases)
    
    prompt = f"""
    {dynamic_instruction}
    
    ELEMEZENDŐ ADATOK:
    {chr(10).join(context_parts)}
    
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
    """

    system_msg = (
        "Te egy tapasztalt, cinikus, de szigorúan objektív politikai és gazdasági elemző vagy. "
        "A feladatod a hírek dekonstrukciója. Ne csak azt nézd, mit írnak, hanem azt is, hogyan. "
        "Keresd a 'keretezési' technikákat és a politikai marketinget. "
        "Légy tömör és lényegretörő. Az egész elemzés ne legyen több 10-12 mondatnál. "
    )

    # Az _gemini_engine hívása marad
    res = _gemini_engine(prompt, system_msg, model_type="lite")
    
    return res if res else "Nem sikerült generálni az elemzést."
    
def get_gemini_embeddings(texts):
    """Vektorok lekérése 100-as csomagokban (Batch limit kezelése)."""
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.models.embed_content(
            model="gemini-embedding-001", # Később érdemes lehet text-embedding-04-re váltani
            contents=batch,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        usage_tracker.add("gemini-embedding-001", response)
        all_embeddings.extend([embedding.values for embedding in response.embeddings])
        if len(texts) > 100:
            time.sleep(1)
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
    
    # Meghívjuk a motort a Lite modellel
    res = _gemini_engine(text, sys_instruct, model_type="lite")
    
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
