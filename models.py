from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from google.genai import Client

class Article(BaseModel):
    id: int
    source: str
    title: str
    summary: str
    link: str
    published: datetime
    embedding: Optional[List[float]] = None

class EventAnalysis(BaseModel):
    """A 'Nagy Flash' által generált mélyelemzés struktúrája."""
    event_title: str = Field(description="Az esemény rövid, beszédes címe")
    summary: str = Field(description="Az esemény objektív összefoglalója")
    discrepancies: List[str] = Field(description="Ténybeli ellentmondások a források között")
    bias_report: Dict[str, str] = Field(description="Forrásonkénti elfogultság és tálalási mód")
    manipulation_index: int = Field(description="1-10 közötti skála a dezinformáció gyanújára")
    article_ids: List[int] = Field(description="Az elemzésben részt vett hírek azonosítói")

class ReportNode(BaseModel):
    """A riport fájának egy csomópontja."""
    title: str # Kategória vagy alkategória neve
    path: List[str] # Teljes útvonal (pl. ["Gazdaság", "Infláció"])
    children: List[Union['ReportNode', EventAnalysis]] = [] # Alkategóriák VAGY Események
    
# Ez a sor kell a Pydantic-nak a ReportNode önhivatkozása miatt
# Pydantic V1/V2 hibrid feloldás
try:
    ReportNode.model_rebuild()
except AttributeError:
    ReportNode.update_forward_refs()

class TokenLogger:
    def __init__(self):
        self.log = []

    def add(self, model_name, response):
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
            
            # Itt is levédjük az összeadást
            stats[m]["in"] += entry.get("input") or 0
            stats[m]["out"] += entry.get("output") or 0
            stats[m]["cached"] += entry.get("cached") or 0
            stats[m]["calls"] += 1
            
            fr = entry.get("finish_reason", "UNKNOWN")
            stats[m]["finish_reasons"][fr] = stats[m]["finish_reasons"].get(fr, 0) + 1
            
        return stats

    def print_summary(self):
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

class SessionContext:
    def __init__(self, articles: Dict[int, Article], cache_id: str, client: Client, token_logger: TokenLogger):
        self.articles = articles
        self.cache_id = cache_id
        self.client = client
        self.logger = token_logger
        self.report_root: Optional[ReportNode] = None
        self.max_depth = 3
        self.config = {"density_high": 0.85, "density_low": 0.65}