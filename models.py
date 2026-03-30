from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

class Article(BaseModel):
    id: int = Field(description="A hír egyedi azonosítója")
    source: str = Field(description="A hír forrása (pl. index, origo)")
    title: str = Field(description="A hír címe")
    summary: str = Field(description="A hír rövid összefoglalója")
    link: str = Field(description="A hír eredeti URL-je")
    tags: List[str] = Field(description="A hírhez kapcsolódó címkék/kategóriák")
    published: datetime = Field(description="A hír megjelenésének időpontja")
    match_score: float = Field(default=0.0, description="A hír relevancia pontszáma a témákhoz (0-1 között)")
    embeddings: List[float] = Field(default_factory=list, description="A hír szövegének numerikus reprezentációja a modellek számára")

    def get_short_text(self) -> str:
        return f"{self.title} - {self.summary[:200]}"

@dataclass
class ClusterNode:
    """
    Egy csomópont a hír-hierarchiában.
    Lehet egy levél (egyedi hír) vagy egy összetett csoport.
    """
    level: int                          # 0: egyedi hír, 1: mikro-csoport, 2: téma...
    centroid: np.ndarray                # A csoport matematikai közepe (normalizált 768d vektor)
    
    # Tartalmi mezők
    summary: str = ""                   # LLM összefoglaló vagy kategória név
    original_text: Optional[str] = None # Csak a 0. szinten (leveleknél) használjuk
    
    # Struktúra mezők
    children: List['ClusterNode'] = field(default_factory=list)
    member_indices: List[int] = field(default_factory=list) # Az eredeti 1000 hír indexei

    def is_leaf(self) -> bool:
        """Visszaadja, hogy a csomópont egyetlen hírt képvisel-e."""
        return self.level == 0

    def get_medoid_child(self) -> 'ClusterNode':
        """
        Kiválasztja azt a közvetlen gyerek-csomópontot, amely 
        matematikailag a legközelebb áll a csomópont közepéhez.
        Ezt a szöveget küldjük majd az LLM-nek reprezentánsként.
        """
        if self.is_leaf() or not self.children:
            return self
            
        # Kinyerjük az összes gyerek centroidját egy mátrixba
        child_centroids = np.array([child.centroid for child in self.children])
        
        # Kiszámoljuk a hasonlóságot a saját centroidunkkal
        similarities = np.dot(child_centroids, self.centroid)
        
        # A legmagasabb hasonlóságú gyerek indexe
        best_idx = int(np.argmax(similarities))
        return self.children[best_idx]




class EventSummaryResult(BaseModel):
    title: str = Field(description="Az esemény rövid, találó magyar neve (pl. 'Kormányinfó: Új adók')")
    summary: str = Field(description="Az esemény átfogó, több forrást szintetizáló összefoglalója")
    category: str = Field(description="Szigorúan csak: HAZAI, GLOBÁLIS vagy EGYÉB")
    score: int = Field(description="Az esemény súlya és fontossága, 1-100 közötti pontszám")


class EventIdCluster(BaseModel):
    ids: List[int] = Field(description="Egy pontosan azonos eseményről szóló hírek ID-jai")

class MultiClusterIdResponse(BaseModel):
    events: List[EventIdCluster] = Field(description="Az azonosított események (klaszterek) listája")

class Scores(BaseModel):
    relevance: int = Field(description="Mennyire kritikus a magyar vagy globális gazdaság/politika szempontjából (1-10)")
    impact: int = Field(description="Az esemény súlya (1-10)")
    novelty: int = Field(description="Mennyire tartalmaz új információt (1-10)")

class ClusterResultSingle(BaseModel):
    name: str = Field(description="Az esemény rövid, magyar neve: SZIGORÚAN MAGYAR NYELVEN, akkor is ha minden forrás angol, csak cégnevek vagy személynevek maradhatnak eredeti formában.")
    ids: List[int] = Field(description="A hírek ID-jai, amik EBBEN az eseményben összeillenek")
    scores: Scores # Itt marad a pontozás (relevance, impact, novelty)
    category: str = Field(description="HAZAI, GLOBÁLIS vagy EGYÉB")

class MultiClusterResponse(BaseModel):
    # Ez fogadja be a laza matematikai csoportot
    events: List[ClusterResultSingle] = Field(description="Az azonosított különálló, releváns események")

class StructuredEventSummary(BaseModel):
    title: str = Field(description="Az esemény rövid, találó magyar neve. SZIGORÚAN MAGYARUL, akkor is ha minden forrás angol, csak cégnevek vagy személynevek maradhatnak eredeti formában.")
    summary: str = Field(description="SZIGORÚAN CSAK A TÉNYEK: Mi történt, kik a szereplők, mik az intézkedések. Semmilyen forráselemzés vagy politikai narratíva nem szerepelhet itt. Fogalmazz lényegretörően, MAXIMUM 500 KARAKTERBEN!")
    left_wing_analysis: str = Field(description="A baloldali/liberális narratíva. Ha nincs ilyen, szigorúan üres string ('') legyen.")
    right_wing_analysis: str = Field(description="A jobboldali/konzervatív narratíva. Ha nincs ilyen, szigorúan üres string ('') legyen.")
    category: str = Field(description="Szigorúan csak: HAZAI, GLOBÁLIS vagy EGYÉB")
    score: int = Field(description="Az esemény fontossága (1-100)")    