from pydantic import BaseModel
from typing import List
from datetime import datetime

class Article(BaseModel):
    id: int
    source: str
    title: str
    summary: str
    link: str
    tags: List[str]
    published: datetime

# Egy adott témához tartozó klaszter modellje
class SingleCluster(BaseModel):
    title: str
    ids: List[int]

# Ezt várjuk az LLM-től a besorolási (Classification) fázisban
class MultiClusterIdResponse(BaseModel):
    events: List[SingleCluster]

# A végleges, HTML-hez használt összefoglaló modellje
class Summary(BaseModel):
    title: str
    summary_text: str
    source_ids: List[int]