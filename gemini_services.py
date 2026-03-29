from typing import List, Dict
import gemini_core
import config
import models

def fill_embeddings(context: models.SessionContext):
    """Lekéri az összes hiányzó vektort a cikkekhez 100-as batch-ekben."""
    articles_to_embed = [a for a in context.articles.values() if a.embedding is None]
    
    for i in range(0, len(articles_to_embed), 100):
        batch = articles_to_embed[i:i+100]
        texts = [f"{a.title}\n{a.summary[:200]}" for a in batch]
        
        res = gemini_core.embed(context, texts)
        for idx, emb in enumerate(res.embeddings):
            batch[idx].embedding = emb.values

def split_and_merge(context: models.SessionContext, article_ids: List[int], path: List[str]) -> Dict[str, List[int]]:
    CHUNK_SIZE = 50
    aggregated_results = {}
    valid_input_ids = set(article_ids)
    
    # 1. KATEGÓRIA STRATÉGIA MEGHATÁROZÁSA
    is_root = len(path) == 0
    if is_root:
        # FIX KATEGÓRIÁK (Első kör)
        base_categories = [
            "Magyar belpolitika és közigazgatás",
            "Magyar gazdaság és üzleti környezet",
            "Globális geopolitika és biztonságpolitika",
            "Világgazdaság és nemzetközi pénzügyek",
            "Vegyes / Egyéb"
        ]
        category_instruction = f"Minden hírt PONTOSAN ezen kategóriák egyikébe sorolj be: {base_categories}. Új kategóriát létrehozni TILOS."
    else:
        # DINAMIKUS KATEGÓRIÁK (További körök)
        base_categories = []
        category_instruction = "Hozz létre 3-4 releváns alkategóriát a hírek tartalma alapján. A kategórianevek magyarok legyenek."

    # 2. CHUNKOK FELDOLGOZÁSA
    for i in range(0, len(article_ids), CHUNK_SIZE):
        chunk = article_ids[i:i + CHUNK_SIZE]
        print(f"      ⏳ Chunk feldolgozás ({i+1}-{min(i+CHUNK_SIZE, len(article_ids))} hír)...")
        
        fragment = "\n".join([f"ID: {aid} | {context.articles[aid].title}" for aid in chunk])
        
        # Frissítjük az instrukciót a már meglévő dinamikus kategóriákkal (ha nem root)
        current_cats = list(aggregated_results.keys())
        if not is_root and current_cats:
            dynamic_instruction = f"Már létező kategóriák: {current_cats}. Elsősorban ezekbe sorolj, csak akkor nyiss újat, ha végképp nem illik bele semmi."
        else:
            dynamic_instruction = ""

        sys_instr = (
            f"Te egy hírszerkesztő vagy. Aktuális szekció: {' > '.join(path) if path else 'Főoldal'}.\n"
            f"{category_instruction}\n{dynamic_instruction}\n"
            "Csak a megadott ID-kat használd!"
        )
        
        prompt = f"Hírek listája:\n{fragment}\n\nOszd be a híreket a JSON struktúra szerint!"

        try:
            response_obj = gemini_core.generate(context, prompt, sys_instr, schema=models.SplitResponse)
            
            if response_obj and hasattr(response_obj, 'buckets'):
                for bucket in response_obj.buckets:
                    cat = bucket.category_name
                    # Csak a bemeneti chunkban szereplő ID-kat fogadjuk el (hallucináció szűrés)
                    clean_ids = [aid for aid in bucket.article_ids if aid in chunk]
                    
                    if clean_ids:
                        if cat not in aggregated_results:
                            # Ha nem root, és az AI új kategóriát talált ki a 3-4 felett, 
                            # itt lehetne korlátozni, de hagyjuk rugalmasan
                            aggregated_results[cat] = []
                        aggregated_results[cat].extend(clean_ids)
                        for cid in clean_ids:
                            valid_input_ids.discard(cid)
        except Exception as e:
            print(f"      ⚠️ Hiba a chunk feldolgozásakor: {e}")

    # 3. MARADÉK KEZELÉSE (Csak az első körben érdekes a Vegyes)
    if valid_input_ids:
        target_cat = "Vegyes / Egyéb" if is_root else f"Egyéb ({path[-1]})"
        print(f"   ⚠️ {len(valid_input_ids)} hír maradt ki a besorolásból -> {target_cat}")
        
        if target_cat not in aggregated_results:
            aggregated_results[target_cat] = []
        aggregated_results[target_cat].extend(list(valid_input_ids))

    return aggregated_results

def llm_anchor_test(context: models.SessionContext, article_ids: List[int], path: List[str]) -> bool:
    fragment = "\n".join([f"ID: {aid} | {context.articles[aid].title}" for aid in article_ids])
    path_str = " > ".join(path) if path else "Gyökér"
    
    sys_instr = f"Kategória: {path_str}. Eldöntendő: Ezek a hírek egyetlen konkrét eseményről szólnak, vagy több különálló témáról?"
    prompt = f"Hírek:\n{fragment}\n\nVálaszolj: 'SINGLE' vagy 'MULTIPLE'."
    
    res_text = gemini_core.generate(context, prompt, sys_instr)
    return "SINGLE" in res_text.upper()

def analyze_event_contrastive(context: models.SessionContext, article_ids: List[int], path: List[str]) -> models.EventAnalysis:
    path_str = " > ".join(path) if path else "Általános"
    prompt = f"Végezz mélyelemzést a hírek alapján (Kategória: {path_str}). Hír-ID-k: {article_ids}. Keresd az ellentmondásokat!"
    sys_instr = "Te egy elfogulatlan oknyomozó újságíró vagy. Használd a kontextusban lévő híreket."
    
    return gemini_core.generate(context, prompt, sys_instr, schema=models.EventAnalysis)