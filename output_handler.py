import telebot
import config
from datetime import datetime

bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def generate_html(final_data_package):
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
            .title {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; text-transform: uppercase; }}
            .summary {{ margin: 15px 0; color: #555; }}
            .sources {{ font-style: italic; font-size: 0.85em; color: #888; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🗞 AI Hírelemzés</h1>
                <p>Frissítve: {now}</p>
            </header>
    """

    categories = [('HAZAI', 'Magyarország'), ('GLOBÁLIS', 'Világhírek'), ('EGYÉB', 'Egyéb')]
    
    for cat_key, cat_label in categories:
        items = [i for i in final_data_package if i['category'] == cat_key]
        if items:
            html_template += f"<h2 class='category-title'>{cat_label}</h2>"
            for item in items:
                html_template += f"""
                <div class="news-card">
                    <span class="score">{item['score']}/10</span>
                    <div class="title">{item['title']}</div>
                    <div class="summary">{item['summary']}</div>
                    <div class="sources">Források: {item['sources']}</div>
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

def process_and_send(final_data_package):
    if not final_data_package:
        print("Nincs küldhető hír.")
        return

    generate_html(final_data_package)

    report_parts = []
    categories = [('HAZAI', 'MAGYARORSZÁG'), ('GLOBÁLIS', 'VILÁGHÍREK'), ('EGYÉB', 'EGYÉB')]

    # Pontszám szerinti sorrend biztosítása a kategóriákon belül is
    final_data_package.sort(key=lambda x: x['score'], reverse=True)

    for cat_key, cat_label in categories:
        items = [i for i in final_data_package if i['category'] == cat_key]
        if items:
            report_parts.append(f"--- {cat_label} ---")
            for item in items:
                # Itt fűzzük hozzá a pontszámot (pl. [8.5/10])
                score_tag = f"[{item['score']}/10]"
                msg = f"📌 {item['title'].upper()} {score_tag}\n\n{item['summary']}\n\n(Forrás: {item['sources']})"
                report_parts.append(msg)

    full_text = "\n\n".join(report_parts)
    send_split_message(config.TELEGRAM_CHAT_ID, full_text)

def send_split_message(chat_id, text):
    MAX_CHARS = 3900
    if len(text) <= MAX_CHARS:
        bot.send_message(chat_id, f"🗞 AI HÍRELEMZÉS\n\n{text}")
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
        header = f"🗞 AI HÍRELEMZÉS ({i}/{total_parts})\n\n"
        bot.send_message(chat_id, header + part)
