import hashlib
import html
import re
from typing import Any, List, TypeVar, Optional, Generic,Dict, List, Set
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, TypeAdapter, BaseModel, Field
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel
from typing import Optional
import json

T = TypeVar('T')

import config 

class NewsItem(BaseModel):
    id: str = Field(description="LLM-barát azonosító (pl. C1)")
    hash: str = "" # Alapértelmezésben üres, a validator tölti ki
    source_id: str
    source_meta: config.RssSource 
    link: str
    published: datetime
    title: str
    content: str
    embedding: Optional[List[float]] = None
    category: str = ""  # Ide kerüljön a "POL", "ECO" stb.
    is_hun: float = 0.0 # Ez maradhat float
    profile: Dict[str, float] = Field(default_factory=dict) # Ebbe NE kerüljön a kategória stringje
    clean_content: Optional[str] = Field(default=None, exclude=True) # Ezt ellenőrizd!
    downloaded: Optional[str] = Field(default=None, exclude=True) # Ezt ellenőrizd!

    def json_for_clustering(self, with_id: bool = False, max_content_width: int = 200) -> str:
        if with_id:
            data = self.model_dump(include={'id', 'title', 'content'})
        else:
            data = self.model_dump(include={'title', 'content'})
        data['content'] = textwrap.shorten(data['content'], width=max_content_width, placeholder="...")
        return json.dumps(data, ensure_ascii=False)

    @field_validator('title', 'content')
    @classmethod
    def validate_cleantext(cls, v: str) -> str:
        return cls.cleantext(v)

    @model_validator(mode='after')
    def compute_hash(self) -> 'NewsItem':
        self.hash = self.generate_news_hash(self.title, self.link)
        return self

    def short_text_for_prompt(self, width: int = 500) -> str:
        combined = f"{self.title} - {self.content}"
        return textwrap.shorten(combined, width=width, placeholder="...")
    
    @classmethod
    def generate_news_hash(cls, title: str, link: str) -> str:
        clean_link = link.split('?')[0].split('#')[0].strip().lower().rstrip('/')
        clean_title = cls.cleantext(title).lower()
        hash_base = f"{clean_title}|{clean_link}"
        return hashlib.sha256(hash_base.encode('utf-8')).hexdigest()
    
    @classmethod
    def cleantext(cls, raw: str) -> str:
        if not raw:
            return ""
        unescaped = html.unescape(raw)
        no_html = re.sub(r'<[^>]+?>', ' ', unescaped)
        return " ".join(no_html.split()).strip()

class NewsCache(BaseModel):
    batches: Dict[str, Dict[str, NewsItem]] = Field(default_factory=dict)
    trash_bin: Dict[str, Set[str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def sync_downloaded_ids(self) -> 'NewsCache':
        for batch_id, items in self.batches.items():
            for item in items.values():
                if not item.downloaded:
                    item.downloaded = batch_id
        return self
    
    def cleanup(self, max_age_hours: int = 24):
        threshold = datetime.now() - timedelta(hours=max_age_hours)
        to_delete = [bid for bid in self.batches.keys() if self._is_too_old(bid, threshold)]
        for bid in to_delete:
            del self.batches[bid]

    @property
    def itemCount(self) -> int:
        return sum(len(batch) for batch in self.batches.values())

class NewsClassification(BaseModel):
    id: str = Field(description="C0, C1...")
    cat: str = Field(description="POL, ECO, TEC, TRASH")
    hun: str = Field(description="HUN or INT")

class BatchClassificationResponse(BaseModel):
    results: List[NewsClassification] = Field(description="A hírek osztályozott listája")


class NewsCluster(BaseModel):
    id: str
    items: List[NewsItem]
    summary_title: str = ""
    is_trash: bool = False
    centroid: Optional[List[float]] = None

    def __init__(self, cluster_id: str, items: List[NewsItem], **data):
        summary = data.get("summary_title", f"Klaszter {cluster_id} ({len(items)} hír)")
        super().__init__(
            id=cluster_id, 
            items=items, 
            summary_title=summary, 
            **data
        )

class LLMClusterResponse(BaseModel):
    id: str = Field(description="A klaszter vagy al-klaszter azonosítója")
    score: int = Field(7, ge=1, le=10, description="Súlyozott pontszám 1 és 10 között")
    title: Optional[str] = Field(None, description="Kulcsszavak listája egy stringben")
    item_ids: List[str] = Field(
        default_factory=list, 
        description="Az eredeti hírek ID-jai (pl. C1, C2)"
    )
    reason: Optional[str] = Field(None, description="Trash esetén a rövid indoklás")

    @property
    def is_trash(self) -> bool:
        return self.score < 7
    

class DualAnchor(BaseModel):
    en: str  # Angol horgony (pl. "US blockade of Hormuz")
    hu: str  # Magyar horgony (pl. "Amerikai blokád a Hormuzi-szorosban")
    en_emb: Optional[List[float]] = Field(default=None, exclude=True)
    hu_emb: Optional[List[float]] = Field(default=None, exclude=True)

class AnchorResponse(BaseModel):
    themes: List[DualAnchor]