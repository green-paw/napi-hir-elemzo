import markdown
import os
from datetime import datetime
from collections import defaultdict
from typing import List, Optional
from models import FinalEvent, ArticleSource

# A GitHub Actions környezeti változója alapján dől el a fájlnév
output_file = os.getenv("OUTPUT_FILENAME", "index.html")

def format_sources_html(sources_list: List[ArticleSource]) -> str:
    """HTML formátumú linkeket gyárt a forrásokból (ArticleSource objektumokból)."""
    source_map = defaultdict(list)
    for s in sources_list:
        source_map[s.name].append(s.url) # Pont-notáció!
    
    formatted = []
    for name, urls in source_map.items():
        if len(urls) == 1:
            formatted.append(f'<a href="{urls[0]}" target="_blank">{name}</a>')
        else:
            links = ", ".join([f'<a href="{url}" target="_blank">{i+1}</a>' for i, url in enumerate(urls)])
            formatted.append(f'{name} ({links})')
    return " | ".join(formatted)

def generate_html(final_data_package: List[FinalEvent], topics_html: str, discarded_summaries: Optional[List[str]] = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Kiszűrjük a kiterjesztést a branch név megjelenítéséhez
    branch_display = output_file.replace("_index.html", "").replace("index.html", "main")
    
    if discarded_summaries is None:
        discarded_summaries = []

    html_template = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Hírelemzés - {now}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; background: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: auto; }}
            header {{ text-align: center; padding: 20px 0; border-bottom: 3px solid #007bff; margin-bottom: 30px; }}
            .category-title {{ background: #007bff; color: white; padding: 10px; border-radius: 5px; margin-top: 40px; text-transform: uppercase; }}
            .news-card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid #28a745; position: relative; }}
            .score {{ float: right; background: #eee; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; font-weight: bold; }}
            .title {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; text-transform: uppercase; margin-bottom: 10px; padding-right: 50px; }}
            .summary {{ margin: 15px 0; color: #444; }}
            .sources {{ font-style: italic; font-size: 0.85em; color: #888; border-top: 1px solid #eee; padding-top: 10px; margin-top: 15px; }}
            .sources a {{ color: #007bff; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🗞 AI Hírelemzés</h1>
                <p>Frissítve: <b>{now}</b> | Verzió: <code>{branch_display}</code></p>
            </header>
    """
    
    if topics_html:
        html_template += f"""
        <div class="strategy-box" style="background: #eef6ff; padding: 15px; border-left: 5px solid #007bff; margin-bottom: 30px; border-radius: 0 8px 8px 0;">
            <h3 style="margin-top: 0; color: #0056b3;">🎯 Napi stratégiai fókuszpontok</h3>
            {topics_html}
        </div>
        """

    # Dinamikus kategória kezelés a csomagban lévő adatok alapján
    present_categories = sorted(list(set(item.category for item in final_data_package)))
    
    for cat in present_categories:
        items = [i for i in final_data_package if i.category == cat]
        if items:
            html_template += f"<h2 class='category-title'>{cat}</h2>"
            for item in items:
                sources_html = format_sources_html(item.sources)
                summary_rendered = markdown.markdown(item.summary)
                
                html_template += f"""
                <div class="news-card">
                    <span class="score">{item.score}</span>
                    <div class="title">{item.title}</div>
                    <div class="summary">{summary_rendered}</div>
                    <div class="sources">Források: {sources_html}</div>
                </div>
                """

    if discarded_summaries:
        html_template += """<hr><div style="color: #666; font-size: 0.9em; padding: 20px; background: #fff; border-radius: 8px;">"""
        html_template += '<h3>🔍 Szűrés során mellőzött kisebb események:</h3><ul>'
        for disc in discarded_summaries:
            html_template += f'<li>{disc}</li>'
        html_template += '</ul></div>'
    
    html_template += "</div></body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"✅ {output_file} sikeresen legyártva.")
    
def process_and_send(final_data_package: List[FinalEvent], topics_html: str, discarded_summaries: Optional[List[str]] = None):
    if not final_data_package:
        print("Nincs küldhető hír.")
        return

    if discarded_summaries is None:
        discarded_summaries = []

    # 1. HTML Generálás
    try:
        generate_html(final_data_package, topics_html, discarded_summaries)
    except Exception as e:
        print(f"❌ Hiba a HTML generálás során: {e}")