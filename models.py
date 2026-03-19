from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Egyetlen hír objektuma (az RSS letöltés után azonnal ilyenné alakítjuk)
class Article(BaseModel):
    id: int
    title: str
    link: str
    source: str
    content: str
    embedding: Optional[List[float]] = None  # Később adjuk hozzá

# 2. A hírforrás objektuma a végső kimenethez
class ArticleSource(BaseModel):
    name: str
    url: str

# 3. A végső, kimenetre szánt esemény (amit a main.py átad az output_handlernek)
class FinalEvent(BaseModel):
    category: str
    title: str
    summary: str
    sources: List[ArticleSource]
    score: int