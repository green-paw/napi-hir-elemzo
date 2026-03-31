import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_MAIN = os.getenv("GOOGLE_API_KEY") # A Tier 1 (fizetős) kulcsod
GOOGLE_API_KEY_FREE = os.getenv("GOOGLE_API_KEY_FREE") # Az ingyenes, nagy limites kulcsod

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"

MODEL_ID = "gemini-2.5-flash"
MODEL_LITE_ID = "gemini-2.5-flash-lite"

from dataclasses import dataclass
from enum import Enum
from typing import Dict

class Language(Enum):
    HU = "hu"
    EN = "en"

class Bias(Enum):
    # Narratíva és irányultság kategóriák az LLM elemzéshez
    OBJECTIVE_AGENCY = "Hírügynökségi / Tényszerű"
    CONSERVATIVE_RIGHT = "Konzervatív / Jobboldali"
    LIBERAL_PROGRESSIVE = "Liberális / Progresszív / Baloldali"
    CRITICAL_INDEPENDENT = "Kritikai / Független"
    STATE_GOVERNMENTAL = "Állami / Kormányzati narratíva"
    EASTERN_STATE = "Keleti / Orosz állami narratíva"
    ECONOMIC_MARKET = "Gazdasági / Piacvezérelt"
    MIXED_AGGREGATOR = "Vegyes / Aggregátor"

@dataclass
class RssSource:
    url: str
    language: Language
    bias: Bias
    description: str

RSS_SOURCES: Dict[str, RssSource] = {
    
    # === NEMZETKÖZI - HÍRÜGYNÖKSÉGI ÉS ÜZLETI ===
    "reuters": RssSource(
        url="https://news.google.com/rss/search?q=site:reuters.com+when:24h&hl=en-US&gl=US&ceid=US:en",
        language=Language.EN,
        bias=Bias.OBJECTIVE_AGENCY,
        description="Globális, objektív hírügynökségi alap"
    ),
    "bloomberg": RssSource(
        url="https://feeds.bloomberg.com/markets/news.rss",
        language=Language.EN,
        bias=Bias.ECONOMIC_MARKET,
        description="Globális üzleti és piaci elemzések"
    ),
    "financial_times": RssSource(
        url="https://news.google.com/rss/search?q=site:ft.com+when:24h&hl=en-US&gl=US&ceid=US:en",
        language=Language.EN,
        bias=Bias.ECONOMIC_MARKET,
        description="Nemzetközi gazdasági folyamatok, mélyebb elemzések"
    ),

    # === NEMZETKÖZI - NYUGATI NARRATÍVÁK (BAL/JOBB) ===
    "fox_news": RssSource(
        url="https://news.google.com/rss/search?q=site:foxnews.com+when:24h&hl=en-US&gl=US&ceid=US:en",
        language=Language.EN,
        bias=Bias.CONSERVATIVE_RIGHT,
        description="USA konzervatív, jobboldali perspektíva"
    ),
    "telegraph": RssSource(
        url="https://news.google.com/rss/search?q=site:telegraph.co.uk+when:24h&hl=en-GB&gl=GB&ceid=GB:en",
        language=Language.EN,
        bias=Bias.CONSERVATIVE_RIGHT,
        description="Brit konzervatív narratíva"
    ),
    "guardian": RssSource(
        url="https://www.theguardian.com/world/rss",
        language=Language.EN,
        bias=Bias.LIBERAL_PROGRESSIVE,
        description="Globális baloldali, liberális nézőpont"
    ),
    "cnn": RssSource(
        url="https://news.google.com/rss/search?q=site:cnn.com+when:24h&hl=en-US&gl=US&ceid=US:en",
        language=Language.EN,
        bias=Bias.LIBERAL_PROGRESSIVE,
        description="USA fősodor, progresszív liberális tálalás"
    ),

    # === NEMZETKÖZI - GEOPOLITIKA ÉS EU ===
    "politico_eu": RssSource(
        url="https://www.politico.eu/feed/",
        language=Language.EN,
        bias=Bias.OBJECTIVE_AGENCY, # Vagy LIBERAL_PROGRESSIVE, témától függően
        description="EU-s döntések, brüsszeli belső infók, magyar-EU viták"
    ),
    "al_jazeera": RssSource(
        url="https://www.aljazeera.com/xml/rss/all.xml",
        language=Language.EN,
        bias=Bias.OBJECTIVE_AGENCY,
        description="Közel-keleti és globális dél perspektívája"
    ),
    "tass_en": RssSource(
        url="https://tass.com/rss/v2.xml",
        language=Language.EN,
        bias=Bias.EASTERN_STATE,
        description="Hivatalos orosz hírügynökségi narratíva (Geopolitikai ellensúly)"
    ),

    # === HAZAI - KORMÁNYZATI ÉS JOBBOLDALI ===
    "hirado_hu": RssSource(
        url="https://hirado.hu/feed/",
        language=Language.HU,
        bias=Bias.STATE_GOVERNMENTAL,
        description="Hivatalos állami média, MTI alapú kormánypárti narratíva"
    ),
    "magyar_nemzet": RssSource(
        url="https://magyarnemzet.hu/feed",
        language=Language.HU,
        bias=Bias.STATE_GOVERNMENTAL,
        description="Kormánypárti véleményformáló és politikai lap"
    ),
    "mandiner": RssSource(
        url="https://mandiner.hu/rss",
        language=Language.HU,
        bias=Bias.CONSERVATIVE_RIGHT,
        description="Hazai, konzervatív, jobboldali fókusz"
    ),

    # === HAZAI - FÜGGETLEN ÉS KRITIKAI (Rovatokra bontva) ===
    "telex_belfold": RssSource(
        url="https://telex.hu/rss/belfold",
        language=Language.HU,
        bias=Bias.CRITICAL_INDEPENDENT,
        description="Telex belföldi hírei (zajszűrt)"
    ),
    "telex_gazdasag": RssSource(
        url="https://telex.hu/rss/gazdasag",
        language=Language.HU,
        bias=Bias.CRITICAL_INDEPENDENT,
        description="Telex makrogazdasági hírei"
    ),
    "24hu_belfold": RssSource(
        url="https://24.hu/belfold/feed/",
        language=Language.HU,
        bias=Bias.CRITICAL_INDEPENDENT,
        description="24.hu hazai politikai és közéleti rovat"
    ),
    "24hu_gazdasag": RssSource(
        url="https://24.hu/fn/feed/",
        language=Language.HU,
        bias=Bias.CRITICAL_INDEPENDENT,
        description="24.hu 'Üzleti' (FN) rovata"
    ),
    "hvg_itthon": RssSource(
        url="https://hvg.hu/rss/itthon",
        language=Language.HU,
        bias=Bias.LIBERAL_PROGRESSIVE,
        description="HVG belföldi politikai és társadalmi hírek"
    ),
    "hvg_gazdasag": RssSource(
        url="https://hvg.hu/rss/gazdasag",
        language=Language.HU,
        bias=Bias.LIBERAL_PROGRESSIVE,
        description="HVG gazdasági elemzések"
    ),
    "444": RssSource(
        url="https://444.hu/feed",
        language=Language.HU,
        bias=Bias.CRITICAL_INDEPENDENT,
        description="Hazai, élesen kritikai, baloldali/liberális tálalás"
    ),

    # === HAZAI - GAZDASÁGI SZAKMAI ===
    "vg": RssSource(
        url="https://www.vg.hu/feed",
        language=Language.HU,
        bias=Bias.ECONOMIC_MARKET,
        description="Világgazdaság - kormánypártibb fókuszú gazdasági elemzések"
    ),
    "portfolio": RssSource(
        url="https://www.portfolio.hu/rss/gazdasag.xml",
        language=Language.HU,
        bias=Bias.ECONOMIC_MARKET,
        description="Szakmai, mély elemzői gazdasági hírportál"
    ),

    # === AGGREGÁTOR ===
    "google_news_hu": RssSource(
        url="https://news.google.com/rss/search?q=hungary+politics+OR+economy&hl=hu&gl=HU&ceid=HU:hu",
        language=Language.HU,
        bias=Bias.MIXED_AGGREGATOR,
        description="Google által válogatott magyar vegyes hírek"
    )
}