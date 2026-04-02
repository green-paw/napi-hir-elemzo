import os
from datetime import datetime
from typing import List

from source import NewsItem

class DebugReporter:
    def __init__(self, output_path: str = "cluster_debug.html"):
        self.output_path = output_path

    def generate(self, macro_clusters: List[List[List["NewsItem"]]], lone_wolves: List["NewsItem"]):
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

        # Makro csoportok listázása
        for i, macro in enumerate(macro_clusters):
            html.append(f"<div class='macro'>")
            html.append(f"<h2># {i+1}. Makró Klaszter ({len(macro)} mikró)</h2>")
            
            for j, micro in enumerate(macro):
                # Az első elem a reprezentatív (vagy centroid közeli)
                rep = micro[0]
                others = micro[1:]
                
                html.append(f"<div class='micro'>")
                html.append(f"<h3>Mikró {j} ({len(micro)} hír)</h3>")
                html.append(f"<div class='rep'>REPREZENTÁNS: {rep.title}</div>")
                html.append(f"<div class='meta'>ID: {rep.id} | Forrás: {rep.source_id}</div>")
                html.append(f"<p>{rep.content[:500]}...</p>")
                
                if others:
                    html.append("<div class='others'><strong>További hírek ebben a mikróban:</strong><ul>")
                    for o in others:
                        html.append(f"<li>{o.title} <span class='meta'>({o.source_id} | {o.id})</span></li>")
                    html.append("</ul></div>")
                
                html.append("</div>")
            html.append("</div>")

        # Magányos hírek
        html.append("<hr><h2>Magányos Hírek (Lone Wolves)</h2><ul>")
        for lw in lone_wolves:
            html.append(f"<li><strong>{lw.title}</strong> <span class='meta'>({lw.source_id} | {lw.id})</span></li>")
        html.append("</ul></body></html>")

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"🔬 Debug HTML kész: {self.output_path}")