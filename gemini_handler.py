import json
import google.genai as genai
from google.genai import types

# Itt inicializáld a klienst (vagy add át paraméterként)
# client = genai.Client(api_key="YOUR_API_KEY")

def _gemini_engine(prompt, sys_instruct, model_type="lite", is_json=False, schema=None):
    """Belső motor a Gemini API hívásokhoz."""
    model_name = "gemini-1.5-flash-lite" if model_type == "lite" else "gemini-1.5-flash"
    
    config = {}
    if is_json:
        config["response_mime_type"] = "application/json"
        if schema:
            config["response_schema"] = schema

    # Megjegyzés: A google.genai (új könyvtár) szintaxisa szerint
    try:
        client = genai.Client(api_key="...") # Vagy használd a globális klienst
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                **config
            )
        )
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini hiba ({model_name}): {e}")
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
    """Lite modell: Klaszterezett adatok validálása és pontozása."""
    sys_instruct = "Válaszd ki az azonos eseményeket, adj nekik magyar címet és kategóriát, majd pontozz."
    return _gemini_engine(cluster_data, sys_instruct, model_type="lite", is_json=True, schema=schema)

def generate_event_summary(event_name, news_contents):
    """Lite modell: Tárgyilagos összefoglaló készítése."""
    sys_instruct = "Írj tárgyilagos, pontos magyar összefoglalót a megadott hírek alapján. Tilos a Markdown!"
    prompt = f"Esemény: {event_name}\n\nForrások:\n{news_contents}"
    res = _gemini_engine(prompt, sys_instruct, model_type="lite", is_json=False)
    return res.strip() if res else "Nem sikerült összefoglalót készíteni."
