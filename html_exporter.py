# html_exporter.py
import datetime
from typing import List
import shared_state
from models import Summary, Article

def get_url_by_id(article_id: int) -> str:
    """Kikeresi egy hír ID-ja alapján az eredeti URL-t."""
    for article in shared_state.filtered_news:
        if article.id == article_id:
            return article.link
    return "#"

def export_to_html(summaries: List[Summary], filename: str = "index.html") -> None:
    """
    A kész összefoglalókból a korábbi letisztult, kártyás stílusú HTML-t generálja.
    """
    current_time: str = datetime.datetime.now().strftime("%Y. %m. %d. %H:%M")
    
    html_content: str = f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Napi Hírösszefoglaló</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --accent: #3498db;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333;
            --secondary-text: #666;
        }}
        body {{ 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
            line-height: 1.6; 
            color: var(--text); 
            background-color: var(--bg); 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 40px 20px; 
        }}
        header {{
            text-align: center;
            margin-bottom: 50px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }}
        h1 {{ color: var(--primary); margin-bottom: 5px; font-weight: 800; }}
        .date {{ color: var(--secondary-text); font-size: 0.9em; }}
        
        .cluster-card {{ 
            background: var(--card-bg); 
            border-radius: 12px; 
            padding: 30px; 
            margin-bottom: 25px; 
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
            border-left: 5px solid var(--accent);
        }}
        .cluster-title {{ 
            color: var(--primary); 
            margin-top: 0; 
            font-size: 1.4em;
            line-height: 1.3;
        }}
        .cluster-summary {{ 
            margin: 15px 0 25px 0; 
            color: #444;
            font-size: 1.05em;
        }}
        .sources-container {{ 
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
        .source-label {{
            font-size: 0.85em;
            color: var(--secondary-text);
            font-weight: 600;
            margin-right: 10px;
        }}
        .source-badge {{ 
            background: #ebf5fb;
            color: var(--accent);
            text-decoration: none;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid #d6eaf8;
        }}
        .source-badge:hover {{ 
            background: var(--accent);
            color: white;
            transform: translateY(-1px);
        }}
        @media (max-width: 600px) {{
            body {{ padding: 20px 15px; }}
            .cluster-card {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Napi Hírösszefoglaló</h1>
        <div class="date">Frissítve: {current_time}</div>
    </header>
"""

    for summary in summaries:
        html_content += f"""
    <div class="cluster-card">
        <h2 class="cluster-title">{summary.title}</h2>
        <div class="cluster-summary">{summary.summary_text}</div>
        <div class="sources-container">
            <span class="source-label">FORRÁSOK:</span>
"""
        # Forrásbadge-ek legenerálása (csak számokkal, mint a képen)
        for i, article_id in enumerate(summary.source_ids):
            url: str = get_url_by_id(article_id)
            html_content += f'<a href="{url}" class="source-badge" target="_blank">#{article_id}</a>'
            
        html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ HTML export kész: {filename}")