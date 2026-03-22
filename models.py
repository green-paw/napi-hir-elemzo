from datetime import datetime

from pydantic import BaseModel, Field
from typing import List

class Article(BaseModel):
    id: int = Field(description="A hír egyedi azonosítója")
    source: str = Field(description="A hír forrása (pl. index, origo)")
    title: str = Field(description="A hír címe")
    summary: str = Field(description="A hír rövid összefoglalója")
    link: str = Field(description="A hír eredeti URL-je")
    tags: List[str] = Field(description="A hírhez kapcsolódó címkék/kategóriák")
    published: datetime = Field(description="A hír megjelenésének időpontja")
    match_score: float = Field(default=0.0, description="A hír relevancia pontszáma a témákhoz (0-1 között)")

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
    name: str = Field(description="Az esemény rövid, magyar neve")
    ids: List[int] = Field(description="A hírek ID-jai, amik EBBEN az eseményben összeillenek")
    scores: Scores # Itt marad a pontozás (relevance, impact, novelty)
    category: str = Field(description="HAZAI, GLOBÁLIS vagy EGYÉB")

class MultiClusterResponse(BaseModel):
    # Ez fogadja be a laza matematikai csoportot
    events: List[ClusterResultSingle] = Field(description="Az azonosított különálló, releváns események")

