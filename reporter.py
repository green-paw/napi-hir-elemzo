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
    print(f"Reporter indítva {cache_obj.itemCount} elemre")
    for batch_id, items_dict in cache_obj.batches.items():
        for news_item in items_dict.values():
            news_item.downloaded = batch_id

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
        <title>News Intelligence Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>"""
    
    html_template += """
    <style>
        /* Alapbeállítások - Világos téma */
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #f8f9fa; 
            color: #212529; 
            margin: 40px auto; 
            max-width: 900px; 
            line-height: 1.5;
        }
        
        h1, h2 { color: #1a1a1a; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
        
        /* Statisztikai doboz */
        .stats { 
            background: #fff; 
            padding: 15px; 
            border: 1px solid #dee2e6; 
            border-radius: 6px; 
            margin-bottom: 30px; 
            display: flex; 
            gap: 30px; 
        }
        .stat-num { font-weight: bold; color: #007bff; }

        /* Kártya stílus - Letisztult fehér */
        .news-card { 
            background: #ffffff; 
            padding: 20px; 
            margin-bottom: 15px; 
            border: 1px solid #dee2e6; 
            border-radius: 8px;
            position: relative;
        }
        
        /* Cím és linkek */
        .news-title { 
            font-weight: bold; 
            font-size: 1.2em; 
            color: #0056b3; 
            text-decoration: none; 
            display: block;
            margin-bottom: 8px;
        }
        .news-title:hover { text-decoration: underline; }
        .news-excerpt { color: #495057; font-size: 0.95em; margin: 10px 0; }

        /* Badge-ek (Kategória és jelzők) */
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            color: #fff;
            margin-right: 5px;
            text-transform: uppercase;
        }
        
        /* Kategória színek - Erőteljes, de tiszta */
        .cat-pol { background-color: #dc3545; } /* Piros */
        .cat-eco { background-color: #28a745; } /* Zöld */
        .cat-tec { background-color: #007bff; } /* Kék */
        .cat-trash { background-color: #6c757d; } /* Szürke */
        
        /* Magyar jelző - Nemzeti színek helyett tiszta keretes stílus */
        .hun-tag { 
            background: #fff; 
            color: #28a745; 
            border: 1px solid #28a745; 
        }

        /* Meta infók (idő, forrás, debug) */
        .card-header { display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center; }
        .source-name { font-weight: bold; color: #343a40; font-size: 0.9em; }
        .timestamp { color: #6c757d; font-size: 0.85em; }
        .debug-info { color: #adb5bd; font-size: 0.7em; margin-top: 10px; border-top: 1px solid #f1f3f5; padding-top: 5px; }

        /* Kategória szerinti szegély a kártya szélén */
        .cat-pol-border { border-left: 6px solid #dc3545; }
        .cat-eco-border { border-left: 6px solid #28a745; }
        .cat-tec-border { border-left: 6px solid #007bff; }
        .cat-trash-border { border-left: 6px solid #6c757d; }
    </style>"""
    
    html_template += f"""
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
            {"".join([_render_card(it, "new") for it in new_items])}
        </div>

        <div class="section">
            <h2>str_cached 💾 CACHE-BŐL</h2>
                {"".join([_render_card(it, "cached") for it in cached_items])}
        </div>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"📊 Riport generálva: {os.path.abspath(filename)}")

def _render_card(item: NewsItem, status: str) -> str:
    # 1. Kategória meghatározása
    cat = getattr(item, 'category', 'OTHER').upper()
    cat_class = f"cat-{cat.lower()}"
    
    # 2. Badge-ek összeállítása
    badges = []
    
    # Kategória badge (mindig az első)
    badges.append(f'<span class="badge {cat_class}">{cat}</span>')
    
    # Magyar vonatkozás
    if item.profile.get("is_hun", 0) > 0.5:
        badges.append('<span class="badge hun-tag">🇭🇺 HUN</span>')
        
    # Relevancia (ha van ilyen adat)
    if item.profile.get("relevance", 0) > 0.8:
        badges.append('<span class="badge high-relevance">🔥 HOT</span>')
    
    badges_html = " ".join(badges)

    # 3. Kártya HTML (javított struktúra)
    return f"""
    <div class="news-card {status} {cat_class}-border">
        <div class="card-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div class="header-left">
                {badges_html}
            </div>
            <div class="header-right" style="font-size: 0.8em; color: #888;">
                <span class="source-name" style="font-weight: bold; margin-right: 10px;">{item.source_id}</span>
                <span class="timestamp">{item.published.strftime('%H:%M')}</span>
            </div>
        </div>
        <div class="card-body">
            <a href="{item.link}" target="_blank" class="title" style="text-decoration: none; display: block; margin-bottom: 5px;">{item.title}</a>
            <div style="font-size: 0.9em; color: #bbb; line-height: 1.4;">
                {item.short_text_for_prompt(200)}
            </div>
        </div>
        <div class="card-footer" style="margin-top: 10px; border-top: 1px solid #333; padding-top: 5px;">
            <small class="debug-info">ID: {item.id} | Hash: {item.hash[:8]}</small>
        </div>
    </div>
    """