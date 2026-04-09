from typing import Any, List, TypeVar, Optional, Generic,Dict, List, Set
from datetime import datetime

T = TypeVar('T')

from pydantic import BaseModel, Field, field_validator, model_validator, TypeAdapter, BaseModel, Field
import textwrap
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import config 

import source

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
        return source.cleantext(v)

    @model_validator(mode='after')
    def compute_hash(self) -> 'NewsItem':
        """Inicializálás után legenerálja a hash-t, ha még nincs."""
        if not self.hash:
            self.hash = source.generate_news_hash(self.title, self.link)
        return self

    def short_text_for_prompt(self, width: int = 500) -> str:
        combined = f"{self.title} - {self.content}"
        return textwrap.shorten(combined, width=width, placeholder="...")

class NewsCache(BaseModel):
    batches: Dict[str, Dict[str, NewsItem]] = Field(default_factory=dict)
    trash_bin: Dict[str, Set[str]] = Field(default_factory=dict)