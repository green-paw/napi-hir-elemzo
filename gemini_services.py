from typing import List, Dict
import gemini_core
import config
import models

def fill_embeddings(context: models.SessionContext):
    """Lekéri az összes hiányzó vektort a cikkekhez 100-as batch-ekben."""
    articles_to_embed = [a for a in context.articles.values() if a.embedding is None]
    
    for i in range(0, len(articles_to_embed), 100):
        batch = articles_to_embed[i:i+100]
        # Csak a címet és a summary elejét küldjük az embeddinghez (költséghatékony)
        texts = [f"{a.title}\n{a.summary[:200]}" for a in batch]
        
        # A gemini_core.embed már a 'gemini-embedding-001'-et használja
        res = gemini_core.embed(context, texts)
        for idx, emb in enumerate(res.embeddings):
            batch[idx].embedding = emb.values

def split_and_merge(context: models.SessionContext, article_ids: List[int], path: List[str]) -> Dict[str, List[int]]:
    fragment = "\n".join([f"ID: {aid} | {context.articles[aid].title}" for aid in article_ids])
    
    sys_instr = f"Te egy hírszerkesztő vagy. Az aktuális útvonalad: {' > '.join(path)}. " \
                "Csoportosítsd a megadott híreket 3-6 releváns alkategóriába!"
    
    prompt = f"Hírek listája:\n{fragment}\n\nOszd be a híreket az objektum struktúra alapján."
    
    # Lekérjük az adatokat a modelltől
    response_obj = gemini_core.generate(context, prompt, sys_instr, schema=models.SplitResponse)
    
    # --- JAVÍTÁS: BIZTONSÁGI HÁLÓ (FALLBACK) ---
    # Ha a response_obj None, vagy a modell valamiért egy sima stringet adott vissza
    if not response_obj or not hasattr(response_obj, 'buckets'):
        print(f"   ⚠️ AI formázási hiba a kategóriák bontásánál! 'Vegyes' csoport alkalmazása.")
        return {"Vegyes (Automatikus)": article_ids}
    
    # Átalakítjuk szótárrá, és kiszűrjük az esetleges üres kategóriákat
    result = {}
    for bucket in response_obj.buckets:
        if bucket.article_ids:  # Csak akkor vesszük fel, ha rakott is bele ID-t
            result[bucket.category_name] = bucket.article_ids
            
    # Még egy utolsó ellenőrzés: ha az AI visszaadott egy objektumot, de tök üresen
    if not result:
        print(f"   ⚠️ Az AI üres kategóriákat generált! 'Vegyes' csoport alkalmazása.")
        return {"Vegyes (Automatikus)": article_ids}
        
    return result

def llm_anchor_test(context: models.SessionContext, article_ids: List[int], path: List[str]) -> bool:
    fragment = "\n".join([f"ID: {aid} | {context.articles[aid].title}" for aid in article_ids])
    
    # ITT HASZNÁLJUK FEL A PATH-T:
    path_str = " > ".join(path) if path else "Általános hírek"
    
    sys_instr = f"Téma kategóriája: {path_str}. Eldöntendő: Ezek a hírek egyetlen konkrét eseményről szólnak, vagy több különálló témáról?"
    prompt = f"Hírek:\n{fragment}\n\nVálaszolj: 'SINGLE' vagy 'MULTIPLE'."
    
    res_text = gemini_core.generate(context, prompt, sys_instr)
    return "SINGLE" in res_text.upper()

def analyze_event_contrastive(context: models.SessionContext, article_ids: List[int], path: List[str]) -> models.EventAnalysis:
    # ITT HASZNÁLJUK FEL A PATH-T:
    path_str = " > ".join(path) if path else "Általános"
    
    prompt = f"Végezz mélyelemzést a következő hír-ID-k alapján: {article_ids}. " \
             f"Vedd figyelembe, hogy ezek a '{path_str}' kategóriába tartoznak! " \
             "Keresd az ellentmondásokat és az elfogultságot!"
    
    sys_instr = "Te egy elfogulatlan oknyomozó újságíró vagy. Használd a kontextusban lévő híreket."
    
    return gemini_core.generate(
        context, 
        prompt, 
        sys_instr, 
        schema=models.EventAnalysis
    )