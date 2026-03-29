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
    
    # --- SZIGORÍTOTT STRATÉGIAI KATEGÓRIÁK ---
    if not path:
        # 0. SZINT: A te pontos kategórialistád
        sys_instr = (
            "Te egy stratégiai hírelemző vagy. A feladatod a hírek PONTOS besorolása az alábbi 5 fő kategóriába:\n"
            "1. 'Magyar belpolitika és közigazgatás'\n"
            "2. 'Magyar gazdaság és üzleti környezet'\n"
            "3. 'Globális geopolitika és biztonságpolitika'\n"
            "4. 'Világgazdaság és nemzetközi pénzügyek'\n"
            "5. 'Vegyes / Egyéb'\n\n"
            "KIZÁRÓLAG ezeket a neveket használd! Minden hírt sorolj be valahová."
        )
    else:
        # MÉLYEBB SZINTEK: Szabadabb, de magyar nyelvű alkategóriák
        sys_instr = (
            f"Te egy szakmai rovatvezető vagy. Aktuális szekció: {' > '.join(path)}. "
            "Bontsd a híreket 3-4 specifikusabb alkategóriára a tartalmuk alapján! "
            "A kategórianevek rövidek és MAGYAR nyelvűek legyenek."
        )
    
    prompt = f"Hírek listája:\n{fragment}\n\nOszd be a híreket a megadott JSON struktúra szerint!"

    # Generálás a Lite modellel
    response_obj = gemini_core.generate(context, prompt, sys_instr, schema=models.SplitResponse)

    # --- BIZTONSÁGI SZŰRÉS (HALLUCINÁCIÓ ELLEN) ---
    if not response_obj or not hasattr(response_obj, 'buckets'):
        return {"Vegyes / Egyéb": article_ids}
    
    result = {}
    valid_input_ids = set(article_ids)
    
    for bucket in response_obj.buckets:
        clean_ids = [aid for aid in bucket.article_ids if aid in valid_input_ids and aid in context.articles]
        if clean_ids:
            result[bucket.category_name] = clean_ids
            for cid in clean_ids: valid_input_ids.remove(cid)
            
    # Ami kimaradt, megy a Vegyesbe
    if valid_input_ids:
        target_cat = "Vegyes / Egyéb" if not path else f"Egyéb ({path[-1]})"
        
        # --- CÍMEK KIÍRÁSA A KONZOLRA ---
        print(f"   ⚠️ {len(valid_input_ids)} hír kimaradt a besorolásból ({target_cat}). Címek:")
        for aid in valid_input_ids:
            title = context.articles[aid].title
            print(f"      - [ID: {aid}] {title}")
        # -------------------------------

        if target_cat in result:
            result[target_cat].extend(list(valid_input_ids))
        else:
            result[target_cat] = list(valid_input_ids)
            
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