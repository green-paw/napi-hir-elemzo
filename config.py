import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"
MODEL_ID = "gemini-2.5-flash" 

RSS_SOURCES = {
    # Nemzetközi források (Angol)
    "reuters": "https://news.google.com/rss/search?q=site:reuters.com+when:24h&hl=en-US&gl=US&ceid=US:en",
    "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",

    # Globális - Konzervatív / Jobboldali (USA/UK)
    "fox_news": "https://news.google.com/rss/search?q=site:foxnews.com+when:24h&hl=en-US&gl=US&ceid=US:en",
    "telegraph": "https://news.google.com/rss/search?q=site:telegraph.co.uk+when:24h&hl=en-GB&gl=GB&ceid=GB:en",

    # Globális - Liberális / Baloldali (USA/UK)
    "guardian": "https://www.theguardian.com/world/rss",
    "cnn": "https://news.google.com/rss/search?q=site:cnn.com+when:24h&hl=en-US&gl=US&ceid=US:en",

    # Globális - Piac és Geopolitika (Kiemelten stabil források)
    "financial_times": "https://news.google.com/rss/search?q=site:ft.com+when:24h&hl=en-US&gl=US&ceid=US:en",
    "al_jazeera": "https://www.aljazeera.com/xml/rss/all.xml"
    
    # Hazai - Jobboldali / Kormánypárti perspektíva
    "magyar_nemzet": "https://magyarnemzet.hu/feed",
    "mandiner": "https://mandiner.hu/rss",

    # Hazai - Baloldali / Liberális / Kritikai perspektíva
    "hvg": "https://hvg.hu/rss",
    "444": "https://444.hu/feed",

    # Hazai - Gazdasági (Alternatív a Portfolio mellé)
    "vg": "https://www.vg.hu/feed"

    # hazai egyéb
    "telex": "https://telex.hu/rss",
    "portfolio": "https://www.portfolio.hu/rss/gazdasag.xml",
    "24hu": "https://24.hu/feed/",
    
    # A jól bevált Google News RSS (Kulcsszavas/Helyi)
    "google_news": "https://news.google.com/rss/search?q=hungary+politics+OR+economy&hl=hu&gl=HU&ceid=HU:hu"
}
