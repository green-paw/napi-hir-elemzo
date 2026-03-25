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
        # Alapértelmezett értékek, ha valami hiba miatt nem jönne metaadat
        entry = {
            "model": model_name,
            "input": 0,
            "output": 0,
            "cached": 0,
            "finish_reason": "UNKNOWN"
        }

        # 1. Költség és token adatok kinyerése
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            entry["input"] = getattr(usage, 'prompt_token_count', 0)
            entry["output"] = getattr(usage, 'candidates_token_count', 0)
            entry["cached"] = getattr(usage, 'cached_content_token_count', 0)

        # 2. Befejezési ok (finish_reason) kinyerése a biztonság/hibakeresés miatt
        if hasattr(response, 'candidates') and response.candidates:
            try:
                fr = response.candidates[0].finish_reason
                # Az enum-ot stringgé alakítjuk a könnyebb olvashatóságért
                entry["finish_reason"] = str(fr).replace('FinishReason.', '') if fr else "UNKNOWN"
            except:
                pass

        self.log.append(entry)

    def get_aggregated_stats(self):
        stats = {}
        for entry in self.log:
            m = entry["model"]
            if m not in stats:
                stats[m] = {
                    "in": 0, 
                    "out": 0, 
                    "cached": 0, 
                    "calls": 0, 
                    "finish_reasons": {}
                }
            
            stats[m]["in"] += entry["input"]
            stats[m]["out"] += entry["output"]
            stats[m]["cached"] += entry["cached"]
            stats[m]["calls"] += 1
            
            fr = entry["finish_reason"]
            stats[m]["finish_reasons"][fr] = stats[m]["finish_reasons"].get(fr, 0) + 1
            
        return stats

    def print_summary(self):
        """Kinyomtatja a futás végi statisztikát a terminálba."""
        stats = self.get_aggregated_stats()
        print("\n" + "="*50)
        print("📊 API HASZNÁLATI ÉS CACHE STATISZTIKA")
        print("="*50)
        
        for model, data in stats.items():
            print(f"🤖 Modell: {model}")
            print(f"   ▶ Hívások száma: {data['calls']}")
            print(f"   📥 Input tokenek:  {data['in']:,} (Ebből Cache: {data['cached']:,})")
            
            # Cache megtakarítás százalékban
            if data['in'] > 0:
                savings = (data['cached'] / data['in']) * 100
                print(f"   ♻️ Cache arány:   {savings:.1f}% megtakarítás")
                
            print(f"   📤 Output tokenek: {data['out']:,}")
            
            # Befejezési okok formázása
            reasons_str = ", ".join([f"{k}: {v}" for k, v in data['finish_reasons'].items()])
            print(f"   🛑 Státuszok:      {reasons_str}")
            print("-" * 50)