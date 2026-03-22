import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_MAIN = os.getenv("GOOGLE_API_KEY") # A Tier 1 (fizetős) kulcsod
GOOGLE_API_KEY_FREE = os.getenv("GOOGLE_API_KEY_FREE") # Az ingyenes, nagy limites kulcsod

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"
MODEL_ID = "gemini-2.5-flash"
MODEL_LITE_ID = "gemini-2.5-flash-lite"

RSS_FEEDS = {
    # Nemzetközi források (Angol) - Hírügynökségi / Üzleti
    "reuters": ("https://news.google.com/rss/search?q=site:reuters.com+when:24h&hl=en-US&gl=US&ceid=US:en", "Nemzetközi, hírügynökségi, objektív"),
    "bloomberg": ("https://feeds.bloomberg.com/markets/news.rss", "Nemzetközi, üzleti, piacvezérelt"),

    # Globális - Konzervatív / Jobboldali (USA/UK)
    "fox_news": ("https://news.google.com/rss/search?q=site:foxnews.com+when:24h&hl=en-US&gl=US&ceid=US:en", "Globális, konzervatív, jobboldali"),
    "telegraph": ("https://news.google.com/rss/search?q=site:telegraph.co.uk+when:24h&hl=en-GB&gl=GB&ceid=GB:en", "Globális, konzervatív, jobboldali"),

    # Globális - Liberális / Baloldali (USA/UK)
    "guardian": ("https://www.theguardian.com/world/rss", "Globális, liberális, baloldali"),
    "cnn": ("https://news.google.com/rss/search?q=site:cnn.com+when:24h&hl=en-US&gl=US&ceid=US:en", "Globális, liberális, progresszív"),

    # Globális - Piac és Geopolitika
    "financial_times": ("https://news.google.com/rss/search?q=site:ft.com+when:24h&hl=en-US&gl=US&ceid=US:en", "Nemzetközi, liberális gazdasági"),
    "al_jazeera": ("https://www.aljazeera.com/xml/rss/all.xml", "Globális, közel-keleti nézőpont"),
    
    # Hazai - Jobboldali / Kormánypárti perspektíva
    "magyar_nemzet": ("https://magyarnemzet.hu/feed", "Hazai, konzervatív, kormánypárti"),
    "mandiner": ("https://mandiner.hu/rss", "Hazai, konzervatív, kormánypárti"),

    # Hazai - Baloldali / Liberális / Kritikai perspektíva
    "hvg": ("https://hvg.hu/rss", "Hazai, liberális, kritikai"),
    "444": ("https://444.hu/feed", "Hazai, baloldali, élesen kritikai"),

    # Hazai - Gazdasági
    "vg": ("https://www.vg.hu/feed", "Hazai, gazdasági, konzervatív"),
    "portfolio": ("https://www.portfolio.hu/rss/gazdasag.xml", "Hazai, gazdasági, elemző"),

    # Hazai egyéb / Független
    "telex": ("https://telex.hu/rss", "Hazai, liberális, független"),
    "24hu": ("https://24.hu/feed/", "Hazai, független, kritikai"),
    
    # Google News RSS (Vegyes)
    "google_news": ("https://news.google.com/rss/search?q=hungary+politics+OR+economy&hl=hu&gl=HU&ceid=HU:hu", "Vegyes hírek, aggregált")
}
