from typing import List, Optional, Any
from models import Article

# A letöltött és előszűrt hírek teljes listája
filtered_news: List[Article] = []

# A véglegesített, LLM által finomhangolt top 30 téma
master_topics: List[str] = []

# A Gemini Cache objektum tárolója. 
# Az 'Any' azért kell ide, hogy ne okozzon import körkörösséget 
# a google.genai könyvtárral, de tudjuk, hogy ez egy Cache objektum lesz.
active_cache: Optional[Any] = None




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