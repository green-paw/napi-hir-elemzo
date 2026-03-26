import re
from tracemalloc import start
import config
import json
from google import genai
from google.genai import types
import time
from typing import Any, Dict, List

from models import Article, EventCluster, MultiClusterIdResponse, MultiClusterResponse
from models import StructuredEventSummary
import llm_core
from models import Article, Topic, LLMTopicList, LLMFilterResponse
from checkpoint_manager import load_checkpoint, save_checkpoint


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



def define_topics(articles: List[Article]) -> List[Topic]:
    topics_state: List[Topic] = load_checkpoint("step1_state.json", List[Topic]) or []
    
    if topics_state and all(t.article_ids for t in topics_state):
        print("✅ Témák betöltve a cache-ből.")
        return topics_state
    
    news_list_str = "\n".join([f"ID:{a.id} | {a.title}" for a in articles])

    sys_instr = f"""
    Te egy vezető geopolitikai, makrogazdasági és nemzetbiztonsági hírszerző elemző vagy. 
    A feladatod a napi hírfolyam dekódolása: a rendszerszintű "jel" elkülönítése a mindennapi "zajtól".

    ALAPELV: Egy hír csak akkor releváns, ha nemzeti, regionális vagy globális szinten befolyásolja a gazdaságot, a politikát vagy a biztonságot.

    KIVÉTEL A BŰNÜGYI SZŰRŐ ALÓL (Ezeket KÖTELEZŐ átengedni):
    - Magas szintű politikai, kormányzati, igazságszolgáltatási vagy nemzetbiztonsági érintettségű büntetőügyek. 
    - Például: hivatali visszaélés, rendszerszintű korrupció, lehallgatási botrányok, magas rangú tisztségviselők (pl. rendőri vezetők, miniszterek) büntetőügyi vallomásai.

    SZIGORÚ TILTÓLISTA (Ezeket AZONNAL és VÉGLEGESEN hagyd figyelmen kívül):
    - KÖZÖNSÉGES bűnügyek, helyi rendőrségi hírek, balesetek (pl. bolti lopás, gyilkosság, közlekedési baleset, drogfogás az utcán).
    - Bulvár, pletyka, celebhírek, magánéleti botrányok.
    - "Kis színes" hírek, érdekességek, furcsaságok, életmód tanácsok.
    - Napi tech-pletykák, új kütyük bejelentése.
    - Egyedi cégek PR-közleményei, kiskereskedelmi akciók (pl. "Drágult a krumpli").
    - Olyan politikai nyilatkozatok, amelyeknek nincs jogi vagy diplomáciai következménye (pl. "X beszólt Y-nak a Facebookon").
    """

    prompt = f"""
    Íme az aktuális nyers hírfolyam:
    {news_list_str}

    ---
    FELADAT:
    A fenti lista alapján (és a tiltólistát szigorúan betartva) határozd meg azt a maximum 5-8 fő STRATÉGIAI IRÁNYVONALAT, ami a mai napot dominálja. 

    SZABÁLYOK A KATEGÓRIÁK LÉTREHOZÁSÁHOZ (A "Vödrök"):
    1. Ne használj túl tág, egy szavas fogalmakat (pl. "Makrogazdaság", "Politika" -> EZEK TILOSAK).
    2. A kategória neve írja le a folyamatot vagy a területet. 
    Jó példák: "Európai biztonságpolitika és NATO-döntések", "Globális inflációs és jegybanki folyamatok", "Magyar költségvetési és adóügyi lépések".
    3. Csak olyan kategóriát hozz létre, amihez van tényleges, súlyos, stratégiai hír a listában. Ha egy nap nincs háborús hír, ne csinálj geopolitikai kategóriát!

    KIMENET:
    Csak a kiválasztott stratégiai irányvonalak neveit add vissza egy JSON listában, semmi mást!
    """

    res = llm_core.gemini_call(
        client=client_main,
        contents=prompt,
        sys_instr=sys_instr,
        model=config.MODEL_LITE_ID,
        schema=LLMTopicList,
        max_output_tokens=1024
    )

    if res and isinstance(res, LLMTopicList):
        topics_state = [Topic(title=t) for t in res.topics]
        save_checkpoint("step1_state.json", topics_state, List[Topic])
        return topics_state
    return []

def gather_articles_for_topics(current_topics: List[Topic], articles: List[Article]) -> List[Topic]:
    topics_state: List[Topic] = load_checkpoint("step1_state.json", List[Topic]) or []
    
    if topics_state and all(t.article_ids for t in topics_state):
        print("✅ Témák betöltve a cache-ből.")
        return topics_state
    
    chunk_size = 100

    sys_instr = """Te egy precíz, gépi adatosztályozó algoritmus vagy. 
    A feladatod egy konkrét stratégiai témakörhöz hozzárendelni a megadott hírlistából a releváns elemeket.

    SZABÁLYOK:
    1. Szigorú egyezés: Csak azokat a híreket válaszd ki, amelyek KÖZVETLENÜL és EGYÉRTELMŰEN az adott témába tartoznak. Ha kétséges, hagyd ki!
    2. Nincs hallucináció: Kizárólag olyan ID-t adhatsz vissza, ami fizikailag szerepel a bemeneti listában.
    3. Formátum: Semmilyen szöveges magyarázatot, bevezetőt ne írj. A válaszod kizárólag egy JSON lista lehet az ID-kkal.
    4. Ha nincs találat, egy üres listát [] kell visszaadnod.
    """

    #loop through topics
    for topic in current_topics:
        #get chunk_size articles at a time
        for i in range(0, len(articles), chunk_size):
            news_chunk = articles[i:i + chunk_size]
            news_chunk_str = "\n".join([f"ID:{a.id} | {a.title}" for a in news_chunk])

            prompt = f"""
            CÉL TÉMAKÖR: 
            "{topic.title}"

            VIZSGÁLANDÓ HÍREK:
            {news_chunk_str}

            ---
            FELADAT:
            Olvasd át a fenti híreket. Válaszd ki KIZÁRÓLAG azoknak a híreknek az ID-ját, amelyek szorosan illeszkednek a "{topic.title}" témakörhöz.
            Add vissza a kiválasztott ID-kat (egész számok) JSON formátumban!
            """

            res = llm_core.gemini_call(
                client=client_main,
                contents=prompt,
                sys_instr=sys_instr,
                model=config.MODEL_LITE_ID,
                schema=list[int],
                max_output_tokens=512
            )

            if res and isinstance(res, list):         
                if topic.article_ids is None:
                    topic.article_ids = []
                topic.article_ids.extend(res)
            elif res and isinstance(res, str):
                try:
                    parsed = json.loads(res)
                    if isinstance(parsed, list):
                        if topic.article_ids is None:
                            topic.article_ids = []
                        topic.article_ids.extend(parsed)
                except:
                    ids = re.findall(r'\b\d+\b', res)
                    if ids:
                        if topic.article_ids is None:
                            topic.article_ids = []
                        topic.article_ids.extend([int(id) for id in ids])

    save_checkpoint("step1_state.json", current_topics, List[Topic])
    return current_topics

def generate_sub_topics(topics: List[Topic], articles: List[Article]) -> List[Topic]:
    topics_state: List[Topic] = load_checkpoint("step2_state.json", List[Topic]) or []
    if topics_state and all(t.events for t in topics_state):
        print("✅ Sub-témák betöltve a cache-ből.")
        return topics_state
    
    # Ha nincs cache, akkor generáljuk újra
    for topic in topics:
        if topic.article_ids is None:
            continue
        
        related_articles = [a for a in articles if a.id in topic.article_ids]
        news_json_str = "\n".join([f"ID: {a.id} | FORRÁS: {a.source} | CÍM: {a.title} | TARTALOM: {a.summary[:300]}" for a in related_articles])

        sys_instr = """Te egy precíz hír-klaszterező algoritmus vagy. 
        A feladatod, hogy egy megadott főtémához tartozó hírlistát konkrét, egyedi ESEMÉNYEK (al-topikok) köré csoportosíts.

        SZABÁLYOK:
        1. Konkrét események: Az al-topik címe legyen nagyon specifikus és leíró (pl. "Kína tajvani hadgyakorlata" és NE az, hogy "Ázsiai feszültség").
        2. Nincs "szemetes" kategória: Ne hozz létre "Egyéb", "Vegyes" vagy "Különféle" nevű eseményeket. Ha egy hír nem kapcsolódik szorosan egy nagyobb eseményhez, hagyd ki.
        3. Precíz ID hozzárendelés: Egy ID csak ahhoz az eseményhez kerülhet be, amiről tényszerűen szól. Nincs hallucináció, csak a bemeneti listában szereplő ID-kat használhatod.
        4. Kimenet: Csak a kért JSON struktúrát add vissza, bevezető és magyarázat nélkül.
        """

        prompt = f"""
        FŐTÉMA: 
        "{topic.title}"

        HÍREK (amelyek ebbe a főtémába tartoznak):
        {news_json_str}

        ---
        FELADAT:
        A fenti híreket csoportosítsd konkrét al-topikokba (eseményekbe). 
        Adj egy specifikus címet az eseménynek SZIGORÚAN MAGYAR NYELVEN, és sorold fel a hozzá tartozó hír ID-kat.
        """
        res = llm_core.gemini_call(
            client=client_main,
            contents=prompt,
            sys_instr=sys_instr,
            model=config.MODEL_LITE_ID,
            schema=List[EventCluster],
            max_output_tokens=2048
        )

        if res and isinstance(res, list):   
            topic.events = res
        else:
            print(f"⚠️ Nem várt válasz az al-témák generálásánál a '{topic.title}' témához. Várható volt egy lista, de ez jött: {res}")
            topic.events = []
    save_checkpoint("step2_state.json", topics, List[Topic])
    return topics        

def process_topics_and_filter(articles: List[Article]) -> List[Topic]:
    topics_state: List[Topic] = load_checkpoint("step1_state.json", List[Topic]) or []
    if topics_state and all(t.article_ids for t in topics_state):
        return topics_state

    needs_save = False

    news_list_str = "\n".join([f"ID:{a.id} | {a.title}" for a in articles])
    
    universal_sys_instr = f"""Te egy vezető stratégiai, politikai és gazdasági elemző AI vagy. 
    A feladatod a zaj kiszűrése és a legfontosabb geopolitikai, makrogazdasági és magyarországi gazdasági vagy politikai események azonosítása a napi hírfolyamból.
    Szigorú fókusz: 
    - Geopolitika és fegyveres konfliktusok
    - Makrogazdaság, nemzetközi piacok, infláció, energia
    - Magyar belpolitika, külpolitika és stratégiai kormányzati döntések
    
    Minden mást (bulvár, sport, napi tech-pletykák, balesetek, időjárás) SZIGORÚAN HAGYJ FIGYELMEN KÍVÜL!
    """

    prompt_base = f"""    
    Íme az aktuális nyers hírfolyam:
    {news_list_str}

    """
    
    # ==========================================
    # LÉPÉS 1: Témák kinyerése (ha még üres a cache)
    # ==========================================
    if not topics_state:
        print("🔄 1. Lépés: Globális témák kinyerése...")
        prompt_1 = prompt_base + """A fenti lista alapján határozd meg azt a maximum 8-10 kiemelt, stratégiailag legfontosabb témát, ami a mai napot dominálja.
        
        Csak a kiválasztott, magas prioritású témák neveit add vissza egy listában (pl. 'Magyar belpolitika', 'Nemzetközi gazdaság').
        Használj JSON formátumot a válaszhoz, csak a témák neveit tartalmazó listával, semmi mással! Ne adj hozzá magyarázatot, ne írj bevezetőt, csak a tiszta lista kell!
        """

        res = llm_core.gemini_call(
            client=client_main,
            contents=prompt_1,
            sys_instr=universal_sys_instr, # <-- BEMENET 1
            model=config.MODEL_LITE_ID,
            schema=LLMTopicList,
            max_output_tokens=1024
        )

        print(res)        

        if res and isinstance(res, LLMTopicList):
            topics_state = [Topic(title=t) for t in res.topics]
            save_checkpoint("step1_state.json", topics_state, List[Topic])

        print("2mp szünet, hátha addig meglesz a cache")
        time.sleep(2)
    
    # ==========================================
    # LÉPÉS 1.5: Hírek ID-jainak besorolása
    # ==========================================
    print("🔄 1.5 Lépés: Hírek keresése az egyes témákhoz (Implicit Cache)...")

    for topic in topics_state:
        if topic.article_ids: 
            continue # Ha ehhez a témához már megvannak az ID-k, ugrunk a következőre
            
        print(f"⏳ Keresés ehhez: '{topic.title}'...")

        chunk_size = 50 # Ez a chunk méret, amit a modellnek küldünk, hogy elkerüljük a token limitet

        #loop through the news list, picking only chunk_size at a time
        for i in range(0, len(articles), chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, len(articles))

            prompt_1_5 = prompt_base + f"""Keresd ki a fenti hírfolyamból azokat a hír ID-kat, amiknek a FŐ TÉMÁJA egyértelműen ez: '{topic.title}'. 

            SZABÁLYOK (Szigorúan tartsd be!):
            1. LÉGY KÍMÉLETLENÜL SZIGORÚ! Csak azt a hírt válaszd ki, ami 100%-ban erről a témáról szól. Ha csak érintőlegesen kapcsolódik, HAGYD KI!
            2. KIZÁRÓLAG egyetlen sornyi, vesszővel elválasztott számsort írj vissza. Ha csak 5 hír illik ide, akkor csak 5 számot adj vissza.
            3. A hírek listájában a számok az ID-k, amik egyértelműen azonosítják a híreket. Ne írd ki a címeket, forrásokat vagy bármi mást, csak a számokat, és semmi mást! Ne adj hozzá magyarázatot, ne írj bevezetőt, csak a tiszta lista kell!
            
            SZIGORÚ UTASÍTÁSOK:
            1. CSAK ÉS KIZÁRÓLAG a(z) {start_idx} és {end_idx} közötti ID-val rendelkező híreket elemezd! 
            2. A listában ezen a tartományon kívül eső összes többi hírt (ID < {start_idx} vagy ID > {end_idx}) TEKINTS SEMMISNEK.
            3. Minden tizedik találatnál ellenőrizd újra, hogy az ID-k valóban a megadott tartományban vannak-e, és ha nem, akkor ne add vissza azokat! Ez a lépés kritikus a pontosság szempontjából, hogy elkerüld a téves besorolást!
            """
            
            res_text = llm_core.gemini_call(
                client=client_main,
                contents=prompt_1_5,
                sys_instr=universal_sys_instr, # <-- UGYANAZ A BEMENET! Itt spórol a Cache.
                model=config.MODEL_LITE_ID,
                schema=None,
                max_output_tokens=512
            )
            
            if res_text and isinstance(res_text, str):
                print(f"{start_idx}-{end_idx} között keresve, talált szöveg: '{res_text}'")
                # A Regex kitép minden egyes számot a szövegből, így az is mindegy, ha a modell véletlenül 
                # azt írná elé, hogy "Íme az ID-k: 12, 34..."
                found_ids: List[int] = [int(num) for num in re.findall(r'\d+', res_text)]
                
                if found_ids:
                    # Eltávolítjuk az esetleges duplikációkat a listából a set() segítségével, és hozzáadjuk a már meglévő id-khoz
                    topic.article_ids = list(set(topic.article_ids) | set(found_ids))
                    print(f"   -> Hozzáadva {len(found_ids)}, összesen: {len(topic.article_ids)} ID.")
                    needs_save = True
                else:
                    print(f"   -> Nem talált ID-kat, vagy üres választ adott.")
    if needs_save:
        save_checkpoint("step1_state.json", topics_state, List[Topic])

    return topics_state
