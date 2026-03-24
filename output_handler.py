import markdown
from datetime import datetime
from collections import defaultdict

import urllib.parse

def generate_ai_search_url(topic_title: str, service: str = "perplexity") -> str:
    """
    Legenerál egy kereső URL-t a megadott AI szolgáltatáshoz.
    """
    # A keresési kifejezés finomítása a jobb találat érdekében
    query = f"Nézz utána ennek a friss eseménynek és foglald össze a részleteket: {topic_title}"
    encoded_query = urllib.parse.quote(query)
    
    urls = {
        "perplexity": f"https://www.perplexity.ai/search?q={encoded_query}",
        "chatgpt": f"https://chatgpt.com/?q={encoded_query}",
        "google": f"https://www.google.com/search?q={encoded_query}",
        "gemini": f"https://gemini.google.com/app?q={encoded_query}"
    }
    
    return urls.get(service, urls["perplexity"])

def format_sources_html(sources_list):
    """HTML formátumú linkeket gyárt a forrásokból."""
    source_map = defaultdict(list)
    for s in sources_list:
        source_map[s['name']].append(s['url'])
    
    formatted = []
    for name, urls in source_map.items():
        if len(urls) == 1:
            formatted.append(f'<a href="{urls[0]}" target="_blank">{name}</a>')
        else:
            links = ", ".join([f'<a href="{url}" target="_blank">{i+1}</a>' for i, url in enumerate(urls)])
            formatted.append(f'{name} ({links})')
    return " | ".join(formatted)

def generate_html(final_data_package, topics_html):
    """Létrehoz egy professzionális, modern HTML fájlt a hírekkel."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Hírelemzés - {now}</title>
        <style>
            :root {{
                --primary: #2563eb;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
                --accent: #10b981;
            }}
            body {{ font-family: 'Inter', -apple-system, sans-serif; line-height: 1.6; background: var(--bg); color: var(--text-main); margin: 0; padding: 20px; }}
            .container {{ max-width: 850px; margin: auto; }}
            header {{ text-align: left; padding: 40px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 30px; }}
            h1 {{ margin: 0; font-size: 2.5rem; letter-spacing: -1px; }}
            .date {{ color: var(--text-muted); font-weight: 500; }}
            
            .category-title {{ 
                font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; 
                color: var(--primary); margin: 40px 0 20px; font-weight: 800;
                display: flex; align-items: center;
            }}
            .category-title::after {{ content: ""; flex: 1; height: 1px; background: #e2e8f0; margin-left: 15px; }}

            .news-card {{ 
                background: var(--card-bg); padding: 25px; margin-bottom: 24px; 
                border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
                transition: transform 0.2s ease; border: 1px solid #f1f5f9;
            }}
            .news-card:hover {{ transform: translateY(-2px); }}
            
            .card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }}
            .score {{ background: #eff6ff; color: var(--primary); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; }}
            .title {{ font-size: 1.4rem; font-weight: 700; color: var(--text-main); line-height: 1.3; flex: 1; padding-right: 15px; }}
            
            .summary {{ margin: 20px 0; color: #334155; font-size: 1.05rem; }}
            .summary ul {{ padding-left: 20px; }}
            
            .footer-row {{ 
                display: flex; justify-content: space-between; align-items: center;
                margin-top: 20px; padding-top: 15px; border-top: 1px solid #f1f5f9;
            }}
            .sources {{ font-size: 0.85rem; color: var(--text-muted); }}
            .sources a {{ color: var(--primary); text-decoration: none; font-weight: 500; }}
            
            .ai-button {{
                display: inline-flex; align-items: center; background: var(--primary); color: white;
                padding: 8px 16px; border-radius: 8px; text-decoration: none; font-size: 0.85rem;
                font-weight: 600; transition: background 0.2s;
            }}
            .ai-button:hover {{ background: #1e40af; }}
            .ai-button svg {{ margin-right: 8px; }}

            .strategy-box {{ background: #ffffff; padding: 20px; border-radius: 12px; border-left: 4px solid var(--accent); box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🗞 Hírelemzés</h1>
                <div class="date">Frissítve: {now}</div>
            </header>
    """
    
    if topics_html:
        html_template += f"""
        <div class="strategy-box">
            <h3 style="margin-top: 0; color: var(--accent);">🎯 Stratégiai fókuszpontok</h3>
            <div style="font-size: 0.95rem;">{topics_html}</div>
        </div>
        """

    categories = [('HAZAI', 'Magyarország'), ('GLOBÁLIS', 'Világhírek'), ('EGYÉB', 'Egyéb')]
    
    for cat_key, cat_label in categories:
        items = [i for i in final_data_package if i['category'] == cat_key]
        if items:
            html_template += f"<div class='category-title'>{cat_label}</div>"
            for item in items:
                sources_html = format_sources_html(item['sources'])
                summary_rendered = markdown.markdown(item['summary'])
                ai_url = generate_ai_search_url(item['title'], "perplexity")

                html_template += f"""
                <div class="news-card">
                    <div class="card-header">
                        <div class="title">{item['title']}</div>
                        <span class="score">{item['score']} / 10</span>
                    </div>
                    <div class="summary">{summary_rendered}</div>
                    <div class="footer-row">
                        <div class="sources">Forrás: {sources_html}</div>
                        <a href="{ai_url}" target="_blank" class="ai-button">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                            Mélyelemzés
                        </a>
                    </div>
                </div>
                """

    html_template += """
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("✅ index.html sikeresen legyártva.")

def process_and_send_old(final_data_package, topics_html):
    if not final_data_package:
        print("Nincs küldhető hír.")
        return

    try:
        generate_html(final_data_package, topics_html)
    except Exception as e:
        print(f"❌ Hiba a HTML generálás során: {e}")

def process_and_send(final_data_package):
    if not final_data_package:
        print("Nincs küldhető adat.")
        return

    now_str = datetime.now().strftime("%Y. %m. %d. %H:%M")
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI News Intelligence Report</title>
        <style>
            :root {{
                --bg-color: #fdfdfb;
                --text-color: #1a1a1a;
                --accent-color: #8b0000; /* Deep red for a classic look */
                --border-color: #d1d1d1;
                --secondary-text: #4a4a4a;
            }}
            body {{
                font-family: 'Georgia', serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                line-height: 1.6;
                margin: 0;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 850px;
                margin: 0 auto;
            }}
            header {{
                border-bottom: 3px solid var(--text-color);
                margin-bottom: 50px;
                padding-bottom: 10px;
                text-align: center;
            }}
            header h1 {{
                font-size: 42px;
                margin: 0;
                text-transform: uppercase;
                letter-spacing: -1px;
                font-weight: 900;
            }}
            .report-meta {{
                font-size: 14px;
                font-family: 'Helvetica', sans-serif;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 10px;
                color: var(--secondary-text);
            }}
            .news-section {{
                margin-bottom: 60px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 40px;
            }}
            .category-tag {{
                font-family: 'Helvetica', sans-serif;
                font-size: 12px;
                font-weight: bold;
                color: var(--accent-color);
                text-transform: uppercase;
                margin-bottom: 10px;
                display: block;
            }}
            h2.article-title {{
                font-size: 32px;
                margin: 0 0 20px 0;
                line-height: 1.1;
                font-weight: bold;
            }}
            .summary-text {{
                font-size: 19px;
                margin-bottom: 30px;
                color: var(--text-color);
                text-align: justify;
            }}
            .analysis-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px double var(--border-color);
            }}
            .analysis-column h3 {{
                font-family: 'Helvetica', sans-serif;
                font-size: 14px;
                text-transform: uppercase;
                margin-top: 0;
                border-bottom: 1px solid var(--text-color);
                padding-bottom: 5px;
            }}
            .analysis-content {{
                font-size: 15px;
                color: var(--secondary-text);
                font-style: italic;
            }}
            .footer-meta {{
                margin-top: 30px;
                font-family: 'Helvetica', sans-serif;
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .source-links a {{
                text-decoration: underline;
            }}
            .deep-dive-link {{
                font-weight: bold;
                color: var(--accent-color);
                text-decoration: none;
                border: 1px solid var(--accent-color);
                padding: 5px 15px;
                transition: all 0.2s;
            }}
            .deep-dive-link:hover {{
                background-color: var(--accent-color);
                color: white;
            }}
            @media (max-width: 600px) {{
                .analysis-grid {{ grid-template-columns: 1fr; }}
                h2.article-title {{ font-size: 26px; }}
            }}
            .category-title {{ 
                font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; 
                color: var(--primary); margin: 40px 0 20px; font-weight: 800;
                display: flex; align-items: center;
            }}
            .category-title::after {{ content: ""; flex: 1; height: 1px; background: #e2e8f0; margin-left: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Intelligence Report</h1>
                <div class="report-meta">Automated Briefing &middot; {now_str}</div>
            </header>
    """

    categories = [('HAZAI', 'Magyarország'), ('GLOBÁLIS', 'Világhírek'), ('EGYÉB', 'Egyéb')]
    for cat_key, cat_label in categories:
        items = [i for i in final_data_package if i['category'] == cat_key]
        if items:
            html_template += f"<div class='category-title'>{cat_label}</div>"

            for item in items:
                summary_html = markdown.markdown(item['summary'])
                ai_url = generate_ai_search_url(item['title'], "perplexity")
                
                # Szétválasztjuk a fő összefoglalót és az elemzéseket a HTML-ben
                # Mivel a main.py-ban összefűztük, itt érdemesebb lenne a summary_data dictet használni,
                # de ha marad a felfűzött verzió, akkor a Markdown parser elvégzi a munkát.
                
                sources_html = format_sources_html(item['sources'])
                #sources_html = ", ".join([f'<a href="{s["url"]}">{s["name"]}</a>' for s in item['sources']])

                html_template += f"""
                <div class="news-section">
                    <span class="category-tag">{item['category']} &nbsp;|&nbsp; Score: {item['score']}</span>
                    <h2 class="article-title">{item['title']}</h2>
                    <div class="summary-text">{summary_html}</div>
                    
                    <div class="footer-meta">
                        <div class="source-links">Források: {sources_html}</div>
                        <a href="{ai_url}" target="_blank" class="deep-dive-link">Perplexity keresés</a>
                    </div>
                </div>
                """

    html_template += """
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("✅ index.html sikeresen legyártva.")