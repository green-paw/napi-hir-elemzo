import markdown
import os
from datetime import datetime
from typing import Any, List, Dict

from models import NewsCache, NewsItem
import urllib.parse
from collections import defaultdict

from gemini_core import logger

def generate_ai_search_url(topic_title: str, service: str = "perplexity") -> str:
    query = f"Nézz utána ennek a friss eseménynek és foglald össze a részleteket: {topic_title}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.perplexity.ai/search?q={encoded_query}"
    return url

def format_sources_html(news: List[NewsItem]) -> str:
    source_map = defaultdict(list)
    for s in news:
        source_map[s.source_id].append(s.link)
    
    formatted = []
    for name, urls in source_map.items():
        if len(urls) == 1:
            formatted.append(f'<a href="{urls[0]}" target="_blank">{name}</a>')
        else:
            links = ", ".join([f'<a href="{url}" target="_blank">{i+1}</a>' for i, url in enumerate(urls)])
            formatted.append(f'{name} ({links})')
    return " | ".join(formatted)

def generate_html_report(cache_obj: NewsCache, current_run_id: str, filename: str = "report.html"):
    # Az aktuális futás hírei (közvetlenül a batch-ből)
    all_items = []
    for rid, batch in cache_obj.batches.items():
        all_items.extend(batch.values())
    all_items.sort(key=lambda x: x.hash)
    
    new_items = list(cache_obj.batches.get(current_run_id, {}).values())

    
    
    # Minden hír, ami NEM a mostani batch-ben van
    cached_items = []
    for rid, batch in cache_obj.batches.items():
        if rid != current_run_id:
            cached_items.extend(batch.values())
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <title>News Intelligence Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #e0e0e0; margin: 40px; }}
            h1, h2 {{ color: #4facfe; }}
            .stats {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 30px; display: flex; gap: 40px; }}
            .stat-box {{ font-size: 1.2em; }}
            .stat-num {{ font-weight: bold; color: #00f2fe; font-size: 1.5em; }}
            .section {{ margin-bottom: 40px; }}
            .news-card {{ background: #252525; padding: 15px; margin-bottom: 10px; border-left: 5px solid #444; border-radius: 4px; }}
            .news-card.new {{ border-left-color: #00c853; }}
            .news-card.cached {{ border-left-color: #ffab00; }}
            .title {{ font-weight: bold; font-size: 1.1em; color: #fff; }}
            .meta {{ color: #888; font-size: 0.9em; margin: 5px 0; }}
            .scores {{ display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }}
            .score-tag {{ background: #3d3d3d; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; border: 1px solid #555; }}
            .high-score {{ background: #4facfe33; border-color: #4facfe; color: #4facfe; font-weight: bold; }}
            .trash-score {{ background: #ff525233; border-color: #ff5252; color: #ff5252; }}
        </style>
    </head>
    <body>
        <h1>📰 News Intelligence Report</h1>
        <p class="meta">{current_run_id}</p>
        <div class="stats">
            <div class="stat-box">Új hírek: <span class="stat-num">{len(new_items)}</span></div>
            <div class="stat-box">Cache-ből: <span class="stat-num">{len(cached_items)}</span></div>
            <div class="stat-box">Összesen: <span class="stat-num">{len(new_items) + len(cached_items)}</span></div>
        </div>

        <div class="section">
            <h2>✨ ÚJ HÍREK</h2>
            {"".join([_render_card(it, "new") for it in all_items])}
        </div>

        <div class="section">
            <h2>str_cached 💾 CACHE-BŐL</h2>
                "".join([_render_card(it, "cached") for it in cached_items])
        </div>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"📊 Riport generálva: {os.path.abspath(filename)}")

def _render_card(it: NewsItem, css_class: str) -> str:
    # Kiválogatjuk a profil értékeket (kivéve a technikai mezőket)
    scores_html = ""
    for k, v in it.profile.items():
        if k == "is_new" or k == "cluster_id": continue
        
        extra_class = "high-score" if v > 0.6 else ""
        if k == "TRASH" and v > 0.7: extra_class = "trash-score"
        
        scores_html += f'<span class="score-tag {extra_class}">{k}: {v:.2f}</span>'

    return f"""
    <div class="news-card {css_class}">
        <div class="title">{it.title}</div>
        <div class="meta">ID: {it.id} | Hash: {it.hash[:10]}...</div>
        <div class="scores">{scores_html}</div>
    </div>
    """