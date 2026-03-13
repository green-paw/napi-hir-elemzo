import telebot
import config

bot = telebot.TeleBot(config.TELEGRAM_TOKEN)

def process_and_send(final_data_package):
    if not final_data_package:
        print("Nincs küldhető hír.")
        return

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
