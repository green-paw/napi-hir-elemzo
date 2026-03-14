import json
import config  # <--- Ezt be kell importálni!
from google import genai
from google.genai import types
import time

# 1. Globális kliens létrehozása itt, a handlerben
client = genai.Client(
    api_key=config.GOOGLE_API_KEY, 
    http_options={'api_version': 'v1beta'}
)

import time
from google.genai import errors

def _gemini_engine(prompt, sys_instruct, model_type="lite", is_json=False, schema=None):
    model_name = "gemini-1.5-flash-lite" if model_type == "lite" else "gemini-1.5-flash"
    
    config_params = {}
    if is_json:
        config_params["response_mime_type"] = "application/json"
        if schema:
            config_params["response_schema"] = schema

    # Újrapróbálkozási logika (maximum 5 kísérlet)
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    temperature=0.0 if is_json else 0.2,
                    **config_params
                )
            )
            
            usage = response.usage_metadata
            print(f"model: {model_name}, input tokens: {usage.prompt_token_count}, output tokens: {usage.candidates_token_count}")
            
            return response.text

        except Exception as e:
            # Ellenőrizzük, hogy 503 (Unavailable) vagy 429 (Rate Limit) hiba történt-e
            error_msg = str(e).lower()
            if "503" in error_msg or "429" in error_msg or "quota" in error_msg:
                wait_time = (attempt + 1) * 5  # Egyre többet vár: 5s, 10s, 15s...
                print(f"⚠️ Szerver túlterhelt (503/429), várakozás {wait_time}s... (Próbálkozás: {attempt+1}/5)")
                time.sleep(wait_time)
            else:
                print(f"❌ Kritikus Gemini hiba ({model_name}): {e}")
                return None
                
    return None

def get_strategic_topics(titles_list):
    """Flash modell: 200+ hír stratégiai átvilágítása."""
    sys_instruct = """Te egy stratégiai hírelemző vagy. A feladatod a 15 legfontosabb téma azonosítása.
    CÉL: Olyan kulcsszavakat kapjak, amikkel később szemantikailag szűrhetjük a hírfolyamot.
    FÓKUSZ: Magyar gazdaság, belpolitika, világgazdaság, háború, technológia, energia.
    VÁLASZ: Csak egy JSON listát adj vissza: ["téma1", "téma2", ...]"""
    
    prompt = f"Elemezd ezeket a címeket a szűrési stratégia alapján:\n{titles_list}"
    res = _gemini_engine(prompt, sys_instruct, model_type="flash", is_json=True)
    try:
        return json.loads(res) if res else []
    except:
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

    SZABÁLY: Ha a hír bulvár jellegű vagy emberi érdekesség (human interest), büntesd alacsony pontszámokkal minden kategóriában!"""

    # Itt hívjuk meg a motort
    res = _gemini_engine(cluster_data, sys_instruct, model_type="lite", is_json=True, schema=schema)
    
    try:
        return json.loads(res) if res else {}
    except Exception as e:
        print(f"⚠️ JSON hiba: {e}")
        return {}

def generate_event_summary(event_name, news_contents):
    """Lite modell: Tárgyilagos összefoglaló készítése."""
    sys_instruct = "Írj tárgyilagos, pontos magyar összefoglalót a megadott hírek alapján. Tilos a Markdown!"
    prompt = f"Esemény: {event_name}\n\nForrások:\n{news_contents}"
    res = _gemini_engine(prompt, sys_instruct, model_type="lite", is_json=False)
    return res.strip() if res else "Nem sikerült összefoglalót készíteni."


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
