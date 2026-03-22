from typing import List, Optional, Any
from models import Article

# A letöltött és előszűrt hírek teljes listája
filtered_news: List[Article] = []

# A véglegesített, LLM által finomhangolt top 30 téma
master_topics: List[str] = []

# A Gemini Cache objektum tárolója. 
# Az 'Any' azért kell ide, hogy ne okozzon import körkörösséget 
# a google.genai könyvtárral, de tudjuk, hogy ez egy Cache objektum lesz.
active_cache: Optional[Any] = None