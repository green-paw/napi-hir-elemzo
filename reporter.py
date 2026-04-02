import os
from datetime import datetime
from typing import Any, List, Dict

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

from clustering import MacroCluster
from source import NewsItem

class DebugReporter:
    def __init__(self, output_path: str = "cluster_debug.html"):
        self.output_path = output_path

    def generate(self, macro_clusters: List[MacroCluster], lone_wolves: List["NewsItem"]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = [
            "<html><head><meta charset='UTF-8'><style>",
            "body { font-family: sans-serif; line-height: 1.5; padding: 20px; color: #000; background: #fff; }",
            ".macro { border: 2px solid #000; margin-bottom: 40px; padding: 20px; }",
            ".micro { border: 1px solid #666; margin: 10px 0 10px 40px; padding: 15px; background: #f9f9f9; }",
            ".rep { font-weight: bold; color: #d00; margin-bottom: 5px; }",
            ".others { font-size: 0.9em; color: #444; border-top: 1px dashed #ccc; margin-top: 10px; padding-top: 5px; }",
            "h1, h2, h3 { margin-top: 0; }",
            ".meta { color: #666; font-size: 0.8em; }",
            "hr { border: 0; border-top: 1px solid #000; margin: 40px 0; }",
            "</style></head><body>",
            f"<h1>Klaszterezés Debug Nézet</h1>",
            f"<p class='meta'>Generálva: {now}</p>",
            f"<p>Összesen: {len(macro_clusters)} makró csoport | {len(lone_wolves)} magányos hír</p>",
            "<hr>"
        ]

        #macro_clusters.sort(key=lambda macro: sum(len(micro) for micro in macro.micro_clusters), reverse=True)

        macro_clusters.sort(
            key=lambda macro: macro.profile["NET_RELEVANCE"],
            reverse=True
        )
        lone_wolves.sort(
            key=lambda item: item.profile["NET_RELEVANCE"],
            reverse=True
        )

        # Makro csoportok listázása
        for i, macro in enumerate(macro_clusters):
            p = macro.profile
            profile_str = f"POL: {p['POLITICS']:.1f} | ECO: {p['ECONOMY']:.1f} | TECH: {p['TECH']:.1f} | TRASH: {p['TRASH']:.1f}"
            
            html.append("<div class='macro'>")
            html.append(f"<b># {i+1} MAKRO ({len(macro.micro_clusters)} mikró)</b>")
            html.append(f"<div class='profile'>PROFIL: {profile_str}</div>")

            macro.micro_clusters.sort(key=len, reverse=True)

            for j, micro in enumerate(macro.micro_clusters):
                html.append(f"<div>Mikró {j} ({len(micro)} hír)</br><ul>")
                for item in micro:
                    html.append(f"<li>{item.title} <span class='meta'>({item.source_id} | {item.id})</span></li>")
                html.append("</ul></div>")


            html.append("</div>")

        html.append("<h2>Lone Wolves (Filtered)</h2><ul>")
        for lw in lone_wolves:
            p = lw.profile
            # Formázott kiíratás a 10-es skálán, fix tizedesekkel
            profile_str = f"POL: {p['POLITICS']:.1f} | ECO: {p['ECONOMY']:.1f} | TECH: {p['TECH']:.1f} | TRASH: {p['TRASH']:.1f}"
            
            html.append(f"<li>")
            html.append(f"<b>{lw.title}</b><br>")
            html.append(f"<small style='color: blue;'>{prof_str}</small>")
            html.append(f"</li>")
        html.append("</ul>")

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"🔬 Debug HTML kész: {self.output_path}")