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
            prof_str = f"P:{p['POLITICS']*10:.1f} | E:{p['ECONOMY']*10:.1f} | T:{p['TECH']*10:.1f} | TR:{p['TRASH']*10:.1f} | NET:{p['NET_RELEVANCE']*10:+.1f}"
            
            html.append(f"<li>")
            html.append(f"<b>{lw.title}</b><br>")
            html.append(f"<small style='color: blue;'>{prof_str}</small>")
            html.append(f"</li>")
        html.append("</ul>")

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"🔬 Debug HTML kész: {self.output_path}")