import markdown
import os
from datetime import datetime
from typing import Any, List, Dict

from models import NewsCache, NewsCluster, NewsItem
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

def generate_html_report(clusters: List[NewsCluster], plot: Any = None, filename: str = "cluster_report.html"):
    """
    Kifejezetten a klaszterek vizualizációjára szolgáló riport.
    """
    print(f"Reporter indítva {len(clusters)} klaszterre")
    
    total_news = sum(len(c.items) for c in clusters)

    plot_html = ""
    if plot:
        plot_html = f"""
        <h2>Klaszter-hierarchia (Condensed Tree)</h2>
        <img src="{{ hdbscan_tree_base64 }}" alt="HDBSCAN Tree" style="max-width: 100%; height: auto;">
        <p><i>A színes ágak jelölik a stabil klasztereket, a szürke ágak a zajt.</i></p>
        """

    html_template = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <title>Klaszter Vizualizáció - {datetime.now().strftime('%H:%M')}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: #f8f9fa; 
                color: #212529; 
                margin: 40px auto; 
                max-width: 1000px; 
                line-height: 1.5;
            }}
            
            h1 {{ color: #1a1a1a; border-bottom: 3px solid #dee2e6; padding-bottom: 10px; }}
            
            .stats {{ 
                background: #fff; 
                padding: 15px; 
                border: 1px solid #dee2e6; 
                border-radius: 6px; 
                margin-bottom: 30px; 
            }}
            
            /* Klaszter Doboz */
            .cluster-box {{ 
                background: #ffffff; 
                padding: 20px; 
                margin-bottom: 25px; 
                border: 1px solid #ced4da; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            
            .cluster-header {{ 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                margin-bottom: 15px; 
                padding-bottom: 10px; 
                border-bottom: 2px solid #e9ecef; 
            }}
            
            .cluster-id {{ font-weight: bold; font-size: 1.3em; color: #007bff; }}
            .cluster-meta {{ color: #6c757d; font-size: 0.9em; }}

            /* Hír kártyák a klaszteren belül */
            .news-item {{ 
                background: #f1f3f5; 
                padding: 10px 15px; 
                margin-bottom: 8px; 
                border-radius: 4px; 
                border-left: 3px solid #adb5bd;
            }}
            .news-title {{ 
                font-weight: bold; 
                color: #495057; 
                text-decoration: none; 
            }}
            .news-title:hover {{ text-decoration: underline; color: #0056b3; }}
            .news-meta {{ color: #868e96; font-size: 0.8em; margin-top: 3px; }}

        </style>
    </head>
    <body>
        <h1>📊 Klaszter Vizualizáció (Threshold Teszt)</h1>
        <div class="stats">
            <div>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div>Összes hír: <span style="font-weight:bold; color:#007bff;">{total_news}</span></div>
            <div>Létrejött klaszterek: <span style="font-weight:bold; color:#007bff;">{len(clusters)}</span></div>
        </div>

        <div class="section">
            {"".join([_render_cluster(c) for c in clusters])}
        </div>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"📊 Klaszter riport generálva: {os.path.abspath(filename)}")

def _render_cluster(cluster: NewsCluster) -> str:
    """Egyetlen klaszter és a benne lévő hírek renderelése."""
    
    # Hírek HTML összeállítása a klaszteren belül
    news_html = ""
    for item in cluster.items:
        news_html += f"""
        <div class="news-item">
            <a href="{item.link}" target="_blank" class="news-title">{item.title}</a>
            <div class="news-meta">
                <span>{item.id}</span> | 
                <span>{item.source_id}</span> | 
                <span>{item.published.strftime('%H:%M')}</span> | 
                <small style="color: #adb5bd">Hash: {item.hash[:8]}</small>
            </div>
        </div>
        """
        
    return f"""
    <div class="cluster-box">
        <div class="cluster-header">
            <div class="cluster-id">{cluster.id} | {"❌" if cluster.is_trash else ""} {cluster.title}</div>
            <div class="cluster-id">{cluster.summary}</div>
            <div class="cluster-meta">
                {len(cluster.items)} hír | Representative: {cluster.items[0].hash[:10]}...
            </div>
        </div>
        <div class="cluster-body">
            {news_html}
        </div>
    </div>
    """