import telebot
import config
import markdown  # pip install markdown
import re
from datetime import datetime
from collections import defaultdict

bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def clean_markdown_for_telegram(text):
    """
    Átalakítja a Markdown formázást Telegram-kompatibilis HTML-re.
    A Telegram HTML parse_mode-ja nem szereti a bonyolult HTML-t, 
    ezért csak a legfontosabbakat alakítjuk át.
    """
    # 1. Vastagítás: **szöveg** -> <b>szöveg</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 2. Listajelek: * vagy - az elején -> • (bullet point)
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)
    
    # 3. Felesleges Markdown maradékok (pl. # címek) eltávolítása vagy formázása
    text = re.sub(r'^#+\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    return text

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

def format_sources_telegram(sources_list):
    """Telegram-kompatibilis HTML linkeket gyárt."""
    source_map = defaultdict(list)
    for s in sources_list:
        source_map[s['name']].append(s['url'])
    
    formatted = []
    for name, urls in source_map.items():
        if len(urls) == 1:
            formatted.append(f'<a href="{urls[0]}">{name}</a>')
        else:
            links = ", ".join([f'<a href="{url}">{i+1}</a>' for i, url in enumerate(urls)])
            formatted.append(f'{name} ({links})')
    return " | ".join(formatted)

def generate_html(final_data_package, topics_html):
    """Létrehoz egy esztétikus HTML fájlt a hírekkel."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
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
            .category-title {{ background: #007bff; color: white; padding: 10px; border-radius: 5px; margin-top: 40px; }}
            .news-card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid #28a745; }}
            .score {{ float: right; background: #eee; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; font-weight: bold; }}
            .title {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; text-transform: uppercase; margin-bottom: 10px; }}
            .summary {{ margin: 15px 0; color: #555; }}
            .summary ul {{ padding-left: 20px; }}
            .sources {{ font-style: italic; font-size: 0.85em; color: #888; border-top: 1px solid #eee; padding-top: 10px; margin-top: 15px; }}
            .sources a {{ color: #007bff; text-decoration: none; }}
            .sources a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🗞 AI Hírelemzés</h1>
                <p>Frissítve: {now}</p>
            </header>
    """
    
    if topics_html:
        html_template += f"""
        <div class="strategy-box" style="background: #f8f9fa; padding: 15px; border-left: 5px solid #007bff; margin-bottom: 30px;">
            <h3 style="margin-top: 0;">🎯 Napi stratégiai fókuszpontok</h3>
            {topics_html}
        </div>
        """

    categories = [('HAZAI', 'Magyarország'), ('GLOBÁLIS', 'Világhírek'), ('EGYÉB', 'Egyéb')]
    
    for cat_key, cat_label in categories:
        items = [i for i in final_data_package if i['category'] == cat_key]
        if items:
            html_template += f"<h2 class='category-title'>{cat_label}</h2>"
            for item in items:
                sources_html = format_sources_html(item['sources'])
                # Markdown átalakítása HTML-re a böngészőhöz
                summary_rendered = markdown.markdown(item['summary'])
                
                html_template += f"""
                <div class="news-card">
                    <span class="score">{item['score']}</span>
                    <div class="title">{item['title']}</div>
                    <div class="summary">{summary_rendered}</div>
                    <div class="sources">Források: {sources_html}</div>
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

def process_and_send(final_data_package, topics_html):
    if not final_data_package:
        print("Nincs küldhető hír.")
        return

    try:
        generate_html(final_data_package, topics_html)
    except Exception as e:
        print(f"❌ Hiba a HTML generálás során: {e}")

    # telegram kihagyása
    return
    
    try:
        report_parts = []
        categories = [('HAZAI', 'MAGYARORSZÁG'), ('GLOBÁLIS', 'VILÁGHÍREK'), ('EGYÉB', 'EGYÉB')]
    
        final_data_package.sort(key=lambda x: x['score'], reverse=True)
    
        for cat_key, cat_label in categories:
            items = [i for i in final_data_package if i['category'] == cat_key]
            if items:
                report_parts.append(f"<b>--- {cat_label} ---</b>")
                for item in items:
                    score_tag = f"<b>[{item['score']}/10]</b>"
                    sources_tg = format_sources_telegram(item['sources'])
                    
                    # Markdown tisztítása a Telegram számára
                    clean_summary = clean_markdown_for_telegram(item['summary'])
                    
                    msg = f"📌 <b>{item['title'].upper()}</b> {score_tag}\n\n{clean_summary}\n\n🔗 <i>Forrás: {sources_tg}</i>"
                    report_parts.append(msg)
    
        full_text = "\n\n".join(report_parts)
        send_split_message(config.TELEGRAM_CHAT_ID, full_text)
    except Exception as e:
        print(f"⚠️ Telegram küldési hiba (de a HTML kész): {e}")        

def send_split_message(chat_id, text):
    MAX_CHARS = 3900
    if len(text) <= MAX_CHARS:
        bot.send_message(chat_id, f"🗞 <b>AI HÍRELEMZÉS</b>\n\n{text}", parse_mode='HTML', disable_web_page_preview=True)
        return

    parts = []
    temp_text = text
    while temp_text:
        if len(temp_text) <= MAX_CHARS:
            parts.append(temp_text.strip())
            break
        split_index = temp_text.rfind('\n\n', 0, MAX_CHARS)
        if split_index == -1: split_index = temp_text.rfind('\n', 0, MAX_CHARS)
        if split_index == -1: split_index = MAX_CHARS
        parts.append(temp_text[:split_index].strip())
        temp_text = temp_text[split_index:].strip()

    total_parts = len(parts)
    for i, part in enumerate(parts, 1):
        header = f"🗞 <b>AI HÍRELEMZÉS ({i}/{total_parts})</b>\n\n"
        bot.send_message(chat_id, header + part, parse_mode='HTML', disable_web_page_preview=True)
