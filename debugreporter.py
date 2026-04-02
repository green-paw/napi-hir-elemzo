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
            key=lambda macro: max(macro.profile["POLITICS"], macro.profile["ECONOMY"], macro.profile["TECH"]) - macro.profile["TRASH"],
            reverse=True
        )

        # Makro csoportok listázása
        for i, macro in enumerate(macro_clusters):
            p = macro.profile
            profile_str = f"POL: {p['POLITICS']} | ECO: {p['ECONOMY']} | TECH: {p['TECH']} | TRASH: {p['TRASH']}"
            
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

        # Magányos hírek
        html.append("<hr><h2>Magányos Hírek (Lone Wolves)</h2><ul>")
        for lw in lone_wolves:
            html.append(f"<li><strong>{lw.title}</strong> <span class='meta'>({lw.source_id} | {lw.id})</span></li>")
        html.append("</ul></body></html>")

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"🔬 Debug HTML kész: {self.output_path}")