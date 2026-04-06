import markdown
import os
from datetime import datetime
from typing import Any, List, Dict

from analyzer import AnalysisResult, MacroAnalysisPair
from output_handler import format_sources_html, generate_ai_search_url

class HtmlReporter:
    """HTML jelentés generálása a validált eseményekből."""
    
    def __init__(self, output_path: str = "index.html"):
        self.output_path = output_path

    def generate(self, events: List[Dict], lone_items: List[Any]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="hu">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Napi Hírfeldolgozó</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 text-gray-900 font-sans">
            <div class="max-w-5xl mx-auto py-10 px-4">
                <header class="mb-12 border-b border-gray-300 pb-6">
                    <h1 class="text-4xl font-bold text-indigo-900">Napi Hírfeldolgozó</h1>
                    <p class="text-gray-600 mt-2">Frissítve: {now}</p>
                </header>

                <section class="mb-12">
                    <h2 class="text-2xl font-semibold mb-6 flex items-center">
                        <span class="bg-indigo-600 text-white px-3 py-1 rounded mr-3">🔥</span> 
                        Kiemelt Események
                    </h2>
                    <div class="grid gap-6">
                        {self._build_event_cards(events)}
                    </div>
                </section>

                <section>
                    <h2 class="text-2xl font-semibold mb-6 flex items-center">
                        <span class="bg-gray-600 text-white px-3 py-1 rounded mr-3">📌</span> 
                        További Fontos Hírek
                    </h2>
                    <div class="bg-white rounded-lg shadow p-6">
                        <ul class="divide-y divide-gray-200">
                            {self._build_lone_items(lone_items)}
                        </ul>
                    </div>
                </section>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"📄 HTML jelentés elkészült: {os.path.abspath(self.output_path)}")

    def _build_event_cards(self, events: List[Dict]) -> str:
        # Rendezés fontosság szerint (10 -> 1)
        sorted_events = sorted(events, key=lambda x: x.get('importance', 0), reverse=True)
        cards = []
        
        for ev in sorted_events:
            importance = ev.get('importance', 5)
            # Színkód az erősség alapján
            bg_color = "bg-red-50" if importance >= 8 else "bg-white"
            border_color = "border-red-200" if importance >= 8 else "border-gray-200"
            
            sources = list(set(item.source_id for item in ev['news_items']))
            
            card = f"""
            <div class="{bg_color} border {border_color} rounded-xl p-6 shadow-sm">
                <div class="flex justify-between items-start mb-4">
                    <span class="text-sm font-bold uppercase tracking-widest text-indigo-600">Relevancia: {importance}/10</span>
                </div>
                <h3 class="text-xl font-bold mb-3 leading-tight text-gray-800">{ev['summary']}</h3>
                <div class="flex flex-wrap gap-2 mt-4">
                    {" ".join([f'<span class="bg-gray-200 text-gray-700 text-xs px-2 py-1 rounded">{s}</span>' for s in sources])}
                </div>
            </div>
            """
            cards.append(card)
        return "\n".join(cards)

    def _build_lone_items(self, lone_items: List[Any]) -> str:
        li_elements = []
        for item in lone_items:
            li = f"""
            <li class="py-3 flex justify-between items-center">
                <span class="text-gray-700 font-medium">{item.title}</span>
                <span class="text-xs text-gray-400 ml-4 italic">{item.source_id}</span>
            </li>
            """
            li_elements.append(li)
        return "\n".join(li_elements)
    

import os
from datetime import datetime
from typing import List

from clustering import MacroCluster, MegaCluster
from source import NewsItem

from datetime import datetime
from typing import List

class DebugReporter:
    def __init__(self, output_path: str = "cluster_debug.html"):
        self.output_path = output_path

    def list_macros(self, macros: List["MacroCluster"]) -> list[str]:
        html = []
        for i, macro in enumerate(macros, 1):
            p = macro.profile
            # Biztonsági .get() hívások, hogy ne omoljon össze, ha egy kulcs hiányzik
            net_rel = p.get('NET_RELEVANCE', 0.0)
            pol = p.get('POLITICS', 0.0)
            eco = p.get('ECONOMY', 0.0)
            tech = p.get('TECH', 0.0)
            trash = p.get('TRASH', 0.0)
            
            profile_str = f"SCORE: {macro.score:.1f} | IMP {macro.impact} NET {net_rel:.1f} | P {pol:.1f} E {eco:.1f} T {tech:.1f} N {trash:.1f}"
            
            html.append("<div class='macro'>")
            html.append(f"<b>#{i} - {macro.title} ({len(macro.micro_clusters)} mikró)</b>")
            html.append(f"<div class='profile'>{profile_str}</div>")

            macro.micro_clusters.sort(key=len, reverse=True)

            for j, micro in enumerate(macro.micro_clusters):
                html.append(f"<div>Mikró {j} ({len(micro)} hír)</br><ul>")
                for item in micro:
                    html.append(f"<li>{item.title} <span class='meta'>({item.source_id} | {item.id})</span></li>")
                html.append("</ul></div>")
            html.append("</div>")
        return html

    def generate(self, mega_clusters: List["MegaCluster"], secondary_macros: List["MacroCluster"], lone_wolves: List["NewsItem"]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_top_macros = sum(len(mega.macros) for mega in mega_clusters)
        total_macros = total_top_macros + len(secondary_macros)
        
        html = [
            "<html><head><meta charset='UTF-8'><style>",
            "body { font-family: sans-serif; line-height: 1.5; padding: 12px; color: #000; background: #fff; }",
            ".mega { border: 3px solid #900; margin-bottom: 20px; padding: 15px; background: #fff8f8; border-radius: 6px; }",
            ".macro { border: 2px solid #000; margin-bottom: 6px; padding: 12px; background: #fff; }",
            ".micro { border: 1px solid #666; margin: 10px 0 10px 40px; padding: 15px; background: #f9f9f9; }",
            ".profile { font-family: monospace; color: #0066cc; font-size: 0.9em; margin: 5px 0; }",
            ".rep { font-weight: bold; color: #d00; margin-bottom: 5px; }",
            ".others { font-size: 0.9em; color: #444; border-top: 1px dashed #ccc; margin-top: 10px; padding-top: 5px; }",
            "h1, h2, h3 { margin-top: 0; }",
            "h3.mega-title { color: #900; }",
            ".meta { color: #666; font-size: 0.8em; }",
            "hr { border: 0; border-top: 1px solid #000; margin: 40px 0; }",
            "</style></head><body>",
            f"<h1>Klaszterezés Debug Nézet (Témakörök)</h1>",
            f"<p class='meta'>Generálva: {now}</p>",
            f"<p>Összesen: {len(mega_clusters)} Mega Klaszter | {total_macros} makró csoport | {len(lone_wolves)} magányos hír</p>",
            "<hr>"
        ]

        # 1. MEGA KLASZTEREK (Kiemelt témakörök)
        html.append(f"<h2>🔥 FŐ TÉMAKÖRÖK (MEGA KLASZTEREK - {len(mega_clusters)} db)</h2>")
        for m_idx, mega in enumerate(mega_clusters, 1):
            html.append("<div class='mega'>")
            # Ha az Editor még nem adott nevet a Megának, egy generikus címet használunk
            mega_title = mega.title if mega.title else f"Témakör #{m_idx}"
            html.append(f"<h3 class='mega-title'>{mega_title} (Átlag Score: {mega.score:.1f} | {len(mega.macros)} Makró)</h3>")
            
            # A makrók kilistázása a témakörön belül
            html.extend(self.list_macros(mega.macros))
            html.append("</div>")

        # 2. MÁSODLAGOS MAKRÓK (Amik nem érték el a Top küszöböt)
        secondary_macros.sort(key=lambda m: m.score, reverse=True)
        html.append(f"<hr><h2>📌 MÁSODLAGOS MAKRÓK ({len(secondary_macros)})</h2>")
        html.extend(self.list_macros(secondary_macros))

        # 3. LONE WOLVES
        lone_wolves.sort(key=lambda item: item.profile.get("NET_RELEVANCE", 0), reverse=True)
        html.append("<hr><h2>🐺 Lone Wolves (Filtered)</h2><ul>")
        for lw in lone_wolves:
            p = lw.profile
            net_rel = p.get('NET_RELEVANCE', 0.0)
            
            html.append(f"<li>")
            html.append(f"<b>{lw.title}</b><br>")
            html.append(f"<small style='color: blue;'>NET: {net_rel:.1f}</small>")
            html.append(f"</li>")
        html.append("</ul>")

        html.append("</body></html>")

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"🔬 Debug HTML kész: {self.output_path}")





def generate_analysis_html(results: List[MacroAnalysisPair], output_file: str = "analysis.html"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = """
    <!DOCTYPE html>
    <html lang="hu">
    <head>
        <meta charset="UTF-8">
        <title>Hírelemzési Jelentés</title>
        <style>
            body { font-family: sans-serif; line-height: 1.6; color: #000; max-width: 800px; margin: 40px auto; padding: 20px; }
            .analysis-card { border-bottom: 2px solid #eee; margin-bottom: 40px; padding-bottom: 20px; }
            .reconstruction { font-weight: 500; margin-bottom: 20px; }
            .meta-section { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #333; margin-top: 15px; }
            .section-title { font-variant: all-small-caps; letter-spacing: 1px; font-weight: bold; margin-top: 10px; display: block; }
            .score { float: right; font-weight: bold; border: 1px solid #000; padding: 2px 8px; }
            .meta { color: #666; font-size: 0.8em; }
            ul { margin: 5px 0; padding-left: 20px; }
            hr { border: 0; border-top: 1px dashed #ccc; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Napi Hírelemzés</h1>
        """
    
    html_content += f"<p class='meta'>Generálva: {now}</p>"
    

    for macro, analysis in results:
        p = macro.profile
        net_rel = p.get('NET_RELEVANCE', 0.0)
        pol = p.get('POLITICS', 0.0)
        eco = p.get('ECONOMY', 0.0)
        tech = p.get('TECH', 0.0)
        trash = p.get('TRASH', 0.0)
        profile_str = f"SCORE: {macro.score:.1f} | IMP {macro.impact} NET {net_rel:.1f} | P {pol:.1f} E {eco:.1f} T {tech:.1f} N {trash:.1f}"
        score = analysis.objectivity_score if analysis.objectivity_score is not None else "N/A"

        links = []
        for micro in macro.micro_clusters:
            for item in micro:
                links.append(item.link)
        sources_html = format_sources_html(links)
        
        html_content += f"""
        <div class="analysis-card">
            <div class="score">Objektivitás: {score}/10</div>
            
            <h3>{macro.title}</h3>
            <p class='meta'>{profile_str}</p>
            <p>
                <p>{markdown.markdown(analysis.reconstruction)}</p>
                
                <span class="section-title">Narratív keretezés</span>
                <p>{markdown.markdown(analysis.narrative_games)}</p>
                
                <span class="section-title">Manipulációs jegyzőkönyv</span>
                <p>{markdown.markdown(analysis.manipulation_log)}</p>
            </p>
            <p class='meta'>{ sources_html }</p>
            <p class='meta'><a href='{ generate_ai_search_url(macro.title) }' target=_BLANK>Perplexity keresés</a></p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Jelentés generálva: {output_file}")