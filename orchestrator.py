import analytics
import gemini_services
from models import ReportNode

def recursive_orchestrator(current_ids, path_nodes, context) -> ReportNode:
    node = ReportNode(title=path_nodes[-1] if path_nodes else "Root", path=path_nodes)
    
    # Báziseset: Ha már elég mélyen vagyunk (pl. 3 szint) vagy kevés a hír
    if len(current_ids) <= 5 or len(path_nodes) >= context.max_depth:
        node.children.append(gemini_services.analyze_event_contrastive(context, current_ids, path_nodes))
        return node

    density = analytics.calculate_density(current_ids, context)
    
    # 1. Nagyon sűrű (egyértelműen egy téma)
    if density >= context.config["density_high"]:
        node.children.append(gemini_services.analyze_event_contrastive(context, current_ids, path_nodes))
        
    # 2. Szürke zóna (Megkérdezzük a Lite modellt, hogy ez egy esemény-e)
    elif context.config["density_low"] <= density < context.config["density_high"]:
        is_single = gemini_services.llm_anchor_test(context, current_ids, path_nodes)
        if is_single:
            node.children.append(gemini_services.analyze_event_contrastive(context, current_ids, path_nodes))
        else:
            _handle_splitting(node, current_ids, path_nodes, context)
            
    # 3. Ritka (Több különböző téma, bontani kell)
    else:
        _handle_splitting(node, current_ids, path_nodes, context)

    return node

def _handle_splitting(node, current_ids, path_nodes, context):
    """Segédfüggvény a vödrözéshez és a rekurzió folytatásához."""
    buckets = gemini_services.split_and_merge(context, current_ids, path_nodes)
    
    for category_name, sub_ids in buckets.items():
        # Csak akkor megyünk tovább, ha az LLM tényleg rakott ID-t a kategóriába
        if sub_ids:
            child_node = recursive_orchestrator(sub_ids, path_nodes + [category_name], context)
            node.children.append(child_node)