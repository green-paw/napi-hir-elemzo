from source import NewsItem
import spacy
import re
from typing import List, Set, Dict

# A modellt a workflow-ban töltjük le, itt csak betöltjük
try:
    nlp = spacy.load("hu_core_news_md")
except OSError:
    import os
    os.system("python -m spacy download hu_core_news_md")
    nlp = spacy.load("hu_core_news_md")

class TextCleaner:
    @staticmethod
    def process(items: List[NewsItem], max_chars: int = 800) -> None:
        for item in items:
            if item.embedding is not None:
                continue

            text = re.sub(r'http\S+', '', item.content)
            text = " ".join(text.split())
            doc = nlp(text)
            entities: Set[str] = {
                ent.text.strip() for ent in doc.ents 
                if ent.label_ in ["PER", "ORG", "GPE"]
            }
            meaningful_words: List[str] = [
                token.text for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space 
                and len(token.text) > 2
            ]
            entity_str = " ".join(list(entities))
            content_str = " ".join(meaningful_words)
            combined_text = f"{item.title} | {entity_str} | {content_str}"

            if not combined_text or len(combined_text) < 50:
                combined_text = f"{item.title} {item.content}"

            item.clean_content = " ".join(combined_text.split())[:max_chars]

        print(f"✨ Szövegtisztítás kész: {len(items)} hír feldolgozva.")