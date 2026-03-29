import markdown
import urllib.parse
from datetime import datetime
from collections import defaultdict
from typing import Union, List
import models

def format_sources_html(articles: List[models.Article]) -> str:
    """HTML formátumú linkeket gyárt a forrásokból a megadott Article objektumok alapján."""
    source_map = defaultdict(list)
    for a in articles:
        source_map[a.source].append(a.link)
    
    formatted = []
    for name, urls in source_map.items():
        if len(urls) == 1:
            formatted.append(f'<a href="{urls[0]}" target="_blank" style="color: #4a5568;">{name}</a>')
        else:
            links = ", ".join([f'<a href="{url}" target="_blank" style="color: #4a5568;">{i+1}</a>' for i, url in enumerate(urls)])
            formatted.append(f'{name} ({links})')
    return " | ".join(formatted)

def _render_node(node: Union[models.ReportNode, models.EventAnalysis], context: models.SessionContext, depth: int = 0) -> str:
    html = ""
    indent = depth * 20 

    if isinstance(node, models.ReportNode):
        if node.title != "Root":
            h_level = min(depth + 1, 4)
            html += f'<div style="margin-left: {indent}px; border-left: 1px solid #eee; padding-left: 15px;">'
            html += f'<h{h_level} class="category-title">{node.title}</h{h_level}>'
        
        for child in node.children:
            html += _render_node(child, context, depth + 1)
        
        if node.title != "Root":
            html += '</div>'

    elif isinstance(node, models.EventAnalysis):
        summary_html = markdown.markdown(node.summary)
        
        # ITT A LÉNYEG: Visszakeressük a cikkeket a context-ből az ID-k alapján
        relevant_articles = [context.articles[aid] for aid in node.article_ids if aid in context.articles]
        sources_html = format_sources_html(relevant_articles)
        
        ai_url = f"https://www.perplexity.ai/search?q={urllib.parse.quote(node.event_title)}"
        
        discrepancies_html = "".join([f"<li>{d}</li>" for d in node.discrepancies])
        bias_html = "".join([f"<li><b>{entry.source}:</b> {entry.description}</li>" for entry in node.bias_report])

        html += f"""
        <div class="news-section" style="margin-left: {indent}px;">
            <span class="category-tag">Elemzés &nbsp;|&nbsp; Manipuláció: {node.manipulation_index}/10</span>
            <h2 class="article-title">{node.event_title}</h2>
            <div class="summary-text">{summary_html}</div>
            
            <div class="analysis-grid">
                <div class="analysis-column">
                    <h3>Ellentmondások</h3>
                    <ul class="analysis-content">{discrepancies_html or "Nincs jelentős eltérés."}</ul>
                </div>
                <div class="analysis-column">
                    <h3>Források torzítása</h3>
                    <ul class="analysis-content">{bias_html}</ul>
                </div>
            </div>

            <div class="footer-meta">
                <div class="source-links">Források: {sources_html}</div>
                <a href="{ai_url}" target="_blank" class="deep-dive-link">AI Mélyfúrás (Perplexity)</a>
            </div>
        </div>
        """
    return html

def create_report(context: models.SessionContext):
    """Létrehozza a végleges index.html fájlt a SessionContext-ben lévő riportból."""
    if not context.report_root:
        print("⚠️ Nincs generált riport a kontextusban.")
        return

    now_str = datetime.now().strftime("%Y. %m. %d. %H:%M")
    
    header_html = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <title>AI News Report</title>
        <style>
            body {{ font-family: 'Georgia', serif; background: #fdfdfb; color: #1a1a1a; padding: 40px; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            header {{ border-bottom: 3px solid #1a1a1a; text-align: center; margin-bottom: 40px; padding-bottom: 20px; }}
            .category-title {{ text-transform: uppercase; letter-spacing: 2px; color: #8b0000; margin-top: 50px; font-family: sans-serif; }}
            .news-section {{ background: white; padding: 30px; border: 1px solid #e2e8f0; margin-bottom: 30px; }}
            .article-title {{ font-size: 28px; font-weight: bold; margin: 0 0 15px 0; }}
            .analysis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; background: #f9f9f9; padding: 15px; margin-top: 20px; border-top: 1px solid #eee; }}
            .analysis-column h3 {{ font-size: 11px; text-transform: uppercase; border-bottom: 1px solid #ccc; font-family: sans-serif; }}
            .analysis-content {{ font-size: 14px; color: #444; padding-left: 20px; font-style: italic; }}
            .footer-meta {{ display: flex; justify-content: space-between; align-items: center; margin-top: 25px; border-top: 1px solid #eee; padding-top: 15px; }}
            .source-links {{ font-size: 12px; font-family: sans-serif; }}
            .deep-dive-link {{ color: #8b0000; font-weight: bold; text-decoration: none; border: 1px solid #8b0000; padding: 5px 12px; font-size: 11px; text-transform: uppercase; font-family: sans-serif; }}
            .category-tag {{ font-size: 10px; color: #999; text-transform: uppercase; font-family: sans-serif; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>AI Hírelemzési Jelentés</h1>
                <div style="font-family: sans-serif; font-size: 12px; letter-spacing: 2px; color: #666;">{now_str}</div>
            </header>
    """

    body_content = _render_node(context.report_root, context)
    
    footer_html = "</div></body></html>"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(header_html + body_content + footer_html)
    print("✅ index.html (linkekkel és hierarchiával) sikeresen legyártva.")