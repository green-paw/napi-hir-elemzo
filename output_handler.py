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
                -font-family: 'Georgia', serif;
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
                -color: var(--accent-color);
                -text-decoration: none;
                -border: 1px solid var(--accent-color);
                -padding: 5px 15px;
                -transition: all 0.2s;
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
                <h1>AI Hírelemzés</h1>
                <div class="report-meta">{now_str}</div>
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

                html_template += f"""
                <div class="news-section">
                    <span class="category-tag">{item['category']} &nbsp;|&nbsp; Score: {item['score']}</span>
                    <h2 class="article-title">{item['title']}</h2>
                    <div class="summary-text">{summary_html}</div>
                    
                    <div class="footer-meta">
                        <div class="source-links">Források: {sources_html}</div>
                        <a href="{ai_url}" target="_blank" class="deep-dive-link">AI websearch</a>
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