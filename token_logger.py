from typing import List, Dict, Any

class TokenLogger:
    def __init__(self) -> None:
        self.log: List[Dict[str, Any]] = []

    def add(self, model_name: str, response: Any) -> None:
        entry = {
            "model": model_name,
            "input": 0,
            "output": 0,
            "cached": 0,
            "finish_reason": "UNKNOWN"
        }

        # 1. Költség és token adatok kinyerése (or 0 védi ki a NoneType hibát)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            entry["input"] = getattr(usage, 'prompt_token_count', 0) or 0
            entry["output"] = getattr(usage, 'candidates_token_count', 0) or 0
            entry["cached"] = getattr(usage, 'cached_content_token_count', 0) or 0

        # 2. Befejezési ok kinyerése
        if hasattr(response, 'candidates') and response.candidates:
            try:
                fr = response.candidates[0].finish_reason
                entry["finish_reason"] = str(fr).replace('FinishReason.', '') if fr else "UNKNOWN"
            except:
                pass

        self.log.append(entry)

    def get_aggregated_stats(self) -> Dict[str, Any]:
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
            
            # Itt is levédjük az összeadást
            stats[m]["in"] += entry.get("input") or 0
            stats[m]["out"] += entry.get("output") or 0
            stats[m]["cached"] += entry.get("cached") or 0
            stats[m]["calls"] += 1
            
            fr = entry.get("finish_reason", "UNKNOWN")
            stats[m]["finish_reasons"][fr] = stats[m]["finish_reasons"].get(fr, 0) + 1
            
        return stats

    def print_summary(self) -> None:
        stats = self.get_aggregated_stats()
        print("-" * 50)
        print("📊 API HASZNÁLATI ÉS CACHE STATISZTIKA")
        print("-" * 50)
        
        for model, data in stats.items():
            print(f"🤖 Modell: {model}")
            print(f"   ▶ Hívások száma: {data['calls']}")
            print(f"   📥 Input tokenek:  {data['in']:,} (Ebből Cache: {data['cached']:,})")
            
            if data['in'] > 0:
                savings = (data['cached'] / data['in']) * 100
                print(f"   ♻️ Cache arány:   {savings:.1f}% megtakarítás")
                
            print(f"   📤 Output tokenek: {data['out']:,}")
            
            reasons_str = ", ".join([f"{k}: {v}" for k, v in data['finish_reasons'].items()])
            print(f"   🛑 Státuszok:      {reasons_str}")
            print("-" * 50)