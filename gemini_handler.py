import config
import json
from google import genai
from google.genai import types
import time
from pydantic import BaseModel, Field
from typing import List

from models import Article, EventSummaryResult, MultiClusterIdResponse, MultiClusterResponse



# 1. Globális kliens létrehozása itt, a handlerben
client_main = genai.Client(api_key=config.GOOGLE_API_KEY)
client_free = genai.Client(api_key=config.GOOGLE_API_KEY_FREE)




class TokenLogger:
    def __init__(self):
        self.log = []

    def add(self, model_name, response):
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            self.log.append({
                "model": model_name,
                # Ha nincs kitöltve, legyen 0
                "input": getattr(usage, 'prompt_token_count', 0),
                "output": getattr(usage, 'candidates_token_count', 0)
            })
        else:
            # Ha egyáltalán nincs metadata (pl. hálózati hiba vagy azonnali tiltás)
            self.log.append({
                "model": model_name,
                "input": 0,
                "output": 0
            })

    def get_aggregated_stats(self):
        try:
            stats = {}
            for entry in self.log:
                m = entry["model"]
                if m not in stats:
                    stats[m] = {"in": 0, "out": 0}
                stats[m]["in"] += entry.get("input", 0)
                stats[m]["out"] += entry.get("output", 0)
            return stats
        except Exception as e:
            print(f"⚠️ Hiba a statisztika összesítésénél: {e}")
            # Hiba esetén visszaadjuk a nyers logot, hogy ne vesszen el adat
            return self.log
        
    def get_summary(self):
        return self.log

# Hozz létre egy példányt a modul szintjén
usage_tracker = TokenLogger()


def _gemini_engine(prompt: str, sys_instruct: str ="Te egy AI asszisztens vagy.", model_type: str="lite", is_json: bool=False, schema=None) -> str:
    model_type = str(model_type).lower() or "lite"

    match model_type:
        case "free":
            current_client = client_free
            current_model = config.MODEL_ID  # gemini-2.5-flash
            max_out = 8192  # A Flash maximuma, hogy biztosan ne vágja le a JSON-t
        case "flash":
            current_client = client_main
            current_model = config.MODEL_ID  # gemini-2.5-flash
            max_out = 2048  # A Lite-nak elég az összefoglalókhoz
        case _:
            current_client = client_main
            current_model = config.MODEL_LITE_ID  # gemini-2.5-flash-lite
            max_out = 2048  # A Lite-nak elég az összefoglalókhoz

    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH")
        for cat in [
            "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_HARASSMENT", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
            "HARM_CATEGORY_DANGEROUS_CONTENT", 
            "HARM_CATEGORY_CIVIC_INTEGRITY"
        ]
    ]
    
    generation_config = {
        "system_instruction": sys_instruct,
        "temperature": 0.0 if is_json else 0.2,
        #"max_output_tokens": max_out,
        "response_mime_type": "application/json" if is_json else "text/plain",        
        "response_schema": schema if is_json and schema else None,
        "safety_settings": safety_settings
    }

    for attempt in range(5):
        try:
            # Itt a kiválasztott current_client-et használjuk!
            response = current_client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(**generation_config)
            )
            if response.candidates[0].finish_reason != "STOP":
                print(f"DEBUG: Finish reason: {response.candidates[0].finish_reason}")

            usage_tracker.add(model_type, response)     
                
            return response.text
        except ValueError as ve:
            # Ha a response.text nem elérhető (pl. teljesen blokkolt tartalom)
            print(f"⚠️ Kimeneti hiba (blokkolt tartalom?): {ve}")
            raise SystemExit(1) 
        except Exception as e:
            error_msg = str(e).lower()
            
            # Csak ezeknél a hibáknál érdemes újrapróbálkozni (Network/Rate limit)
            retry_errors = ["503", "429", "quota", "overloaded", "unavailable"]
            
            if any(err in error_msg for err in retry_errors):
                wait_time = (attempt + 1) * 5
                print(f"⚠️ Szerver túlterhelt, várakozás {wait_time}s... ({attempt+1}/5)")
                time.sleep(wait_time)
            else:
                print(f"Hiba a Gemini motorban ({model_type}): {e}")
                raise SystemExit(1) 

    print(f"Hiba a Gemini motorban ({model_type}): 5 retry után sem sikerült választ kapni.")
    raise SystemExit(1) 

def _gemini_engine_old(prompt, sys_instruct, model_type="lite", is_json=False, schema=None):
    model_name = "gemini-2.5-flash-lite"
    if model_type == "flash":
        model_name = "gemini-2.5-flash"

    if model_name == "gemini-2.5-flash" and is_json == True and schema == None:
        print("WARNING: nincs megadva séma egy flash json hívásnál")

    # Újrapróbálkozási logika (maximum 5 kísérlet)
    for attempt in range(5):
        try:
            response = client_main.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    temperature=0.0 if is_json else 0.2,
                    response_mime_type="application/json" if is_json else "text/plain",
                    response_schema=schema if is_json and schema else None,
                    max_output_tokens=2048,
                    safety_settings = [
                        types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH")
                        for cat in [
                            "HARM_CATEGORY_HATE_SPEECH", 
                            "HARM_CATEGORY_HARASSMENT", 
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                            "HARM_CATEGORY_DANGEROUS_CONTENT", 
                            "HARM_CATEGORY_CIVIC_INTEGRITY"
                        ]
                    ]
                )
            )

            has_valid_content = (
                response.candidates and 
                response.candidates[0].content and 
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].text
            )

            # A _gemini_engine függvényben a generálás után:
            if response.candidates[0].finish_reason != "STOP":
                print(f"DEBUG: Finish reason: {response.candidates[0].finish_reason}")
                
                # Ha tiltás van, írjuk ki a részleteket
                if response.candidates[0].finish_reason == "SAFETY" or response.candidates[0].finish_reason == "PROHIBITED_CONTENT":
                    print("--- Biztonsági szűrések részletei ---")
                    for rating in response.candidates[0].safety_ratings:
                        # Csak azokat írjuk ki, amik nem 'NEGLIGIBLE' (elhanyagolható) szintűek
                        if rating.probability != "NEGLIGIBLE":
                            print(f"Kategória: {rating.category} | Valószínűség: {rating.probability} | Prompt: {prompt}")
    
            usage_tracker.add(model_name, response)            

            # Ellenőrizzük, hogy félbeszakadt-e a válasz
            if response.candidates[0].finish_reason == "MAX_TOKENS":
                print(f"⚠️ FIGYELMEZTETÉS ({model_name}): A válasz túl hosszú, le lett vágva!")
            
            # Csak akkor próbálunk JSON-t varázsolni, ha azt kértük
            if is_json:
                try:
                    # Megpróbáljuk leszedni a Markdown kódrészleteket, ha a modell odatette volna
                    cleaned_text = response.text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text.replace("```", "", 1).rsplit("```", 1)[0].strip()
                    
                    return cleaned_text # Visszaadjuk a szöveget a hívónak, ő fogja json.loads-olni
                except Exception as e:
                    print(f"❌ Hiba a JSON szöveg előkészítésénél: {e}")
                    return None
            
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

    # nem sikerült 5 próbálkozás alatt sem
    print(f"❌ KRITIKUS HIBA ({model_name}): nem sikerült 5 próbálkozás alatt sem")
    raise SystemExit(1)
    return None

def get_strategic_topics(titles_sample):
    prompt = f"""
    Elemezd a következő hírcímeket, és határozz meg maximum 7 darab kiemelt stratégiai fókuszpontot, amelyek a mai napot dominálják.

    Szigorú prioritási sorrend:
    - Geopolitika és Háború: Katonai konfliktusok, eszkaláció, nemzetközi szankciók, Irán, Ukrajna, USA-Kína feszültség.
    - Magyar Stratégiai Érdek: Hazai belpolitikai válságok, választási kampány, kormányzati döntések, nemzetbiztonság.
    - Kritikus Gazdaság és Energia: Infláció, forint-összeomlás, energiabiztonság, olajárak, nagyvállalati (OTP, MOL, CATL) krízishelyzetek.
    
    Tiltólista:
    - Csak akkor említs technológiát, klímát vagy kultúrát, ha az közvetlen, súlyos gazdasági vagy politikai következménnyel jár (pl. AI-szabályozás miatti tőzsdei bukás).
    - Ha nincs 7 valóban stratégiai téma, adj kevesebbet, de ne töltsd fel bulvárral vagy irreleváns "trendekkel".
    
    Formátum: Csak a témák címeit add vissza, fontossági sorrendben.

    HÍREK:
    {titles_sample}
    
    VÁLASZ FORMÁTUMA:
    Egy JSON listát adj vissza, ami csak a témák nevét tartalmazza, semmi mást!
    Példa: ["Téma 1", "Téma 2", "Téma 3"]
    """
    
    res_text = str(_gemini_engine(
        prompt=prompt, 
        sys_instruct="Te egy stratégiai politikai, gazdasági elemző vagy.",
        is_json=True,
        schema=list[str] 
    ))

    try:
        import json
        topics = json.loads(res_text)
        if isinstance(topics, list):
            return topics
        return []
    except Exception as e:
        print(f"❌ Hiba a témák feldolgozásánál: {e}")
        print(res_text)
        return []

def validate_news_clusters(cluster_data, schema=MultiClusterResponse):
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
    res = _gemini_engine(cluster_data, sys_instruct, model_type="lite", is_json=True, schema=schema)
    
    try:
        if not res: return {"events": []}
        data = json.loads(res)
        # Biztosítjuk, hogy mindig legyen egy 'events' kulcsunk
        return data if "events" in data else {"events": []}
    except Exception as e:
        print(f"⚠️ JSON hiba a validációnál: {e}")
        return {"events": []}

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

    res = _gemini_engine(prompt, system_msg)
    
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
    
    res = _gemini_engine(text, sys_instruct)
    
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
        response = _gemini_engine(
            prompt,
            sys_instruct=sys_instruct,
            model_type="lite",
            is_json=True,
            schema=MultiClusterIdResponse
        )
        return json.loads(response)
    except Exception as e:
        print(f"Hiba a Flash klaszterezésnél: {e}")
        return MultiClusterIdResponse(events=[])