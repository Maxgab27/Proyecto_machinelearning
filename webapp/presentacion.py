import json
import csv
from pathlib import Path


def dashboard(folder: Path):
    if not (folder/'COMPLETADO.json').is_file():
        raise FileNotFoundError('No hay un análisis completo disponible.')
    summary = json.loads((folder/'resumen.json').read_text(encoding='utf-8'))
    with (folder/'datos_preparados.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    # Reducir tamaño del panel, conservando extremos y fecha final.
    stride = max(1, len(rows)//280)
    selected = rows[::stride]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    series = [{'date': r['Date'][:10], 'close': float(r['close']),
               'ma7': float(r['Media_Movil_7']), 'ma30': float(r['Media_Movil_30'])} for r in selected]
    with (folder/'nlp_terminos.csv').open(encoding='utf-8', newline='') as handle:
        terms = list(csv.DictReader(handle))[:15]
    with (folder/'predicciones.csv').open(encoding='utf-8', newline='') as handle:
        predictions = list(csv.DictReader(handle))[-20:]
    return {'summary': summary, 'series': series, 'terms': terms,
            'preview': rows[:8], 'predictions': predictions,
            'graphs': sorted(p.name for p in (folder/'graficos').glob('*.png'))}
