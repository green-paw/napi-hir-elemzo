from models import NewsItem
import re
from typing import List, Set, Dict

class TextCleaner:
    # Kombinált stopword lista (magyar + angol)
    STOPWORDS = {
        # Magyar
        'a', 'az', 'egy', 'és', 'vagy', 'hogy', 'nem', 'is', 'be', 'ki', 'le', 'fel', 
        'meg', 'el', 'át', 'volna', 'lett', 'volt', 'még', 'már', 'csak', 'mert', 
        'mint', 'után', 'alatt', 'között', 'amikor', 'vagyis', 'tehát', 'hiszen',
        'vagyok', 'vagy', 'vagyunk', 'vagytok', 'vannak', 'lesz', 'lett',
        
        # Angol (leggyakoribbak)
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
        'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they'
    }

    @staticmethod
    def process(items: List['NewsItem'], max_chars: int = 800) -> None:
        for item in items:
            if item.embedding is not None:
                continue
            raw_text = f"{item.title} {item.content}"
            text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.lower().split()
            filtered_words = [w for w in words if w not in TextCleaner.STOPWORDS and len(w) > 2]
            item.clean_content = " ".join(filtered_words)[:max_chars]
        print(f"✨ Szövegtisztítás kész: {len(items)} hír feldolgozva.")

    @staticmethod
    def process_single(item: NewsItem, max_chars: int = 800) -> None:
        if item.embedding is not None:
            return
        raw_text = f"{item.title} {item.content}"
        text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.lower().split()
        filtered_words = [w for w in words if w not in TextCleaner.STOPWORDS and len(w) > 2]
        item.clean_content = " ".join(filtered_words)[:max_chars]
