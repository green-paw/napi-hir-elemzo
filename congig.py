GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HISTORY_FILE = "history.json"
MODEL_ID = "gemini-2.5-flash" 

RSS_SOURCES = {
    # Nemzetközi források (Angol)
    "reuters": "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best",
    "bloomberg": "https://www.bloomberg.com/feeds/bview/rss",
    
    # Hazai források (Magyar)
    "telex": "https://telex.hu/rss",
    "portfolio": "https://www.portfolio.hu/rss/gazdasag.xml",
    "24hu": "https://24.hu/feed/",
    
    # A jól bevált Google News RSS (Kulcsszavas/Helyi)
    "google_news": "https://news.google.com/rss/search?q=hungary+politics+OR+economy&hl=hu&gl=HU&ceid=HU:hu"
}
