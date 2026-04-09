import hashlib
import html
import re
from typing import Any, List, TypeVar, Optional, Generic,Dict, List, Set
from datetime import datetime

T = TypeVar('T')

from pydantic import BaseModel, Field, field_validator, model_validator, TypeAdapter, BaseModel, Field
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

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
    clean_content: Optional[str] = Field(default=None, exclude=True) # Ezt ellenőrizd!
    profile: Dict[str, float] = Field(default_factory=dict)

    @field_validator('title', 'content')
    @classmethod
    def validate_cleantext(cls, v: str) -> str:
        return cleantext(v)

    @model_validator(mode='after')
    def compute_hash(self) -> 'NewsItem':
        self.hash = generate_news_hash(self.title, self.link)
        return self

    def short_text_for_prompt(self, width: int = 500) -> str:
        combined = f"{self.title} - {self.content}"
        return textwrap.shorten(combined, width=width, placeholder="...")

class NewsCache(BaseModel):
    batches: Dict[str, Dict[str, NewsItem]] = Field(default_factory=dict)
    trash_bin: Dict[str, Set[str]] = Field(default_factory=dict)




def generate_news_hash(title: str, link: str) -> str:
    """Stabil SHA-256 hasht generál a cím és a tisztított link alapján."""
    # 1. Link drasztikusabb tisztítása (trailing slash eltávolítása is)
    clean_link = link.split('?')[0].split('#')[0].strip().lower().rstrip('/')
    
    # 2. Cím tisztítása (ugyanazt a logikát használva, mint a modell)
    clean_title = cleantext(title).lower()
    
    hash_base = f"{clean_title}|{clean_link}"
    return hashlib.sha256(hash_base.encode('utf-8')).hexdigest()

def cleantext(raw: str) -> str:
    """HTML mentesítés, entitás dekódolás és whitespace normalizálás."""
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    # Tagek cseréje szóközre (hogy ne ragadjanak össze a szavak)
    no_html = re.sub(r'<[^>]+?>', ' ', unescaped)
    return " ".join(no_html.split()).strip()