import analytics
import gemini_services
from models import ReportNode

def recursive_orchestrator(current_ids, path_nodes, context) -> ReportNode:
    # --- 1. VIZUÁLIS LOGOLÁS: Hol tartunk a fában? ---
    depth = len(path_nodes)
    indent = "  " * depth # A mélységtől függő szóközök
    path_str = " > ".join(path_nodes) if path_nodes else "Root (Minden hír)"
    
    print(f"{indent}📂 [{depth}. szint] {path_str} | Hírek: {len(current_ids)} db")
    
    node = ReportNode(title=path_nodes[-1] if path_nodes else "Root", path=path_nodes)
    
    # --- 2. BÁZISESET (Végállomás) ---
    if len(current_ids) <= 5 or len(path_nodes) >= context.max_depth:
        print(f"{indent}   ⚡ Báziseset elérve (Max mélység vagy kevés hír). Elemzés generálása...")
        analysis = gemini_services.analyze_event_contrastive(context, current_ids, path_nodes)
        node.children.append(analysis)
        print(f"{indent}   ✅ Elemzés kész: {analysis.event_title}")
        return node

    # --- 3. MATEK: Sűrűség vizsgálat ---
    density = analytics.calculate_density(current_ids, context)
    print(f"{indent}   📊 Sűrűség: {density:.2f}", end=" -> ")
    
    # A) Nagyon sűrű (egyértelműen egy téma)
    if density >= context.config["density_high"]:
        print("🎯 Magas. Egyetlen esemény elemzése...")
        analysis = gemini_services.analyze_event_contrastive(context, current_ids, path_nodes)
        node.children.append(analysis)
        print(f"{indent}   ✅ Elemzés kész: {analysis.event_title}")
        
    # B) Szürke zóna (Megkérdezzük a Lite modellt)
    elif context.config["density_low"] <= density < context.config["density_high"]:
        print("🤔 Szürke zóna. Horgony-teszt (LLM) indítása...")
        is_single = gemini_services.llm_anchor_test(context, current_ids, path_nodes)
        
        if is_single:
            print(f"{indent}   ⚓ Teszt: SINGLE (Egy téma). Elemzés generálása...")
            analysis = gemini_services.analyze_event_contrastive(context, current_ids, path_nodes)
            node.children.append(analysis)
            print(f"{indent}   ✅ Elemzés kész: {analysis.event_title}")
        else:
            print(f"{indent}   ⚓ Teszt: MULTIPLE (Több téma). Bontás következik...")
            _handle_splitting(node, current_ids, path_nodes, context, indent)
            
    # C) Ritka (Több különböző téma, fixen bontani kell)
    else:
        print("🔀 Alacsony. Hírek szétválogatása alkategóriákra...")
        _handle_splitting(node, current_ids, path_nodes, context, indent)

    return node

def _handle_splitting(node, current_ids, path_nodes, context, indent):
    """Segédfüggvény a vödrözéshez és a rekurzió folytatásához."""
    buckets = gemini_services.split_and_merge(context, current_ids, path_nodes)
    
    print(f"{indent}   ✂️ {len(buckets)} új alkategória jött létre: {list(buckets.keys())}")
    
    for category_name, sub_ids in buckets.items():
        if sub_ids:
            # --- RÖVIDZÁR: A Vegyes kategóriát elobjuk ---
            if "Vegyes" in category_name:
                print(f"{indent}   🗑️ '{category_name}' ág átugrása (nem pazarolunk rá AI elemzést).")
                continue # Ugrik a következő kategóriára, az AI hívás elmarad
                
            # Normál kategóriák rekurzív hívása
            child_node = recursive_orchestrator(sub_ids, path_nodes + [category_name], context)
            node.children.append(child_node)