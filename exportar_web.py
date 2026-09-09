"""Exporta únicamente resultados de consulta al frontend desplegable en Vercel."""
import json
import shutil
from pathlib import Path
from webapp.presentacion import dashboard

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    source = ROOT/'resultados'
    destination = ROOT/'frontend'/'public'/'demo'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'dashboard.json').write_text(json.dumps(dashboard(source), ensure_ascii=False), encoding='utf-8')
    for name in ['INFORME_TECNICO.html','metricas.csv','predicciones.csv','nlp_terminos.csv','nlp_paginas.csv']:
        shutil.copy2(source/name, destination/name)
    shutil.copytree(source/'graficos', destination/'graficos', dirs_exist_ok=True)
    print(f'Resultados exportados a {destination}')
