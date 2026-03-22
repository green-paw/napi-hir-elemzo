# html_exporter.py
import datetime
from typing import List
import shared_state
from models import Summary, Article

def get_url_by_id(article_id: int) -> str:
    """Kikeresi egy hír ID-ja alapján az eredeti URL-t a globális memóriából."""
    for article in shared_state.filtered_news:
        if article.id == article_id:
            return article.link
    return "#"

def export_to_html(summaries: List[Summary], filename: str = "napi_hirek.html") -> None:
    """
    A kész összefoglalókból egy formázott, reszponzív HTML fájlt generál.
    """
    current_time: str = datetime.datetime.now().strftime("%Y. %m. %d. %H:%M")
    
    html_content: str = f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Napi Hírösszefoglaló</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f4f7f6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        .date {{ text-align: center; color: #7f8c8d; margin-bottom: 40px; font-size: 0.9em; }}
        .cluster-card {{ background: #fff; border-radius: 8px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .cluster-title {{ color: #2980b9; margin-top: 0; }}
        .cluster-summary {{ margin-bottom: 20px; white-space: pre-wrap; }}
        .sources {{ font-size: 0.85em; border-top: 1px solid #eee; padding-top: 15px; }}
        .sources a {{ color: #3498db; text-decoration: none; margin-right: 15px; display: inline-block; margin-bottom: 5px; }}
        .sources a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Napi Hírösszefoglaló</h1>
    <div class="date">Frissítve: {current_time}</div>
"""

    for summary in summaries:
        html_content += f"""
    <div class="cluster-card">
        <h2 class="cluster-title">{summary.title}</h2>
        <div class="cluster-summary">{summary.summary_text}</div>
        <div class="sources">
            <strong>Források: </strong>
"""
        # Forráslinkek legenerálása
        for article_id in summary.source_ids:
            url: str = get_url_by_id(article_id)
            if url != "#":
                html_content += f'<a href="{url}" target="_blank">🔗 Cikk #{article_id}</a>'

        html_content += """
        </div>
    </div>"""

    html_content += """
</body>
</html>"""

    # Fájlba írás
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"🌐 HTML sikeresen legenerálva és elmentve: {filename}")