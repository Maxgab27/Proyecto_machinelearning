import json
import csv
from collections import deque
from itertools import islice
from pathlib import Path


def dashboard(folder: Path):
    if not (folder/'COMPLETADO.json').is_file():
        raise FileNotFoundError('No hay un análisis completo disponible.')
    summary = json.loads((folder/'resumen.json').read_text(encoding='utf-8'))
    with (folder/'datos_preparados.csv').open(encoding='utf-8', newline='') as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    # Reducir tamaño del panel, conservando extremos y fecha final.
    stride = max(1, count//280)
    selected, preview = [], []
    with (folder/'datos_preparados.csv').open(encoding='utf-8', newline='') as handle:
        last = None
        for i, row in enumerate(csv.DictReader(handle)):
            if i < 8:
                preview.append(row)
            if i % stride == 0:
                selected.append(row)
            last = row
        if last is not None and (count-1) % stride:
            selected.append(last)
    series = [{'date': r['Date'][:10], 'close': float(r['close']),
               'ma7': float(r['Media_Movil_7']), 'ma30': float(r['Media_Movil_30'])} for r in selected]
    with (folder/'nlp_terminos.csv').open(encoding='utf-8', newline='') as handle:
        terms = list(islice(csv.DictReader(handle), 15))
    with (folder/'predicciones.csv').open(encoding='utf-8', newline='') as handle:
        predictions = list(deque(csv.DictReader(handle), maxlen=20))
    return {'summary': summary, 'series': series, 'terms': terms,
            'preview': preview, 'predictions': predictions,
            'graphs': sorted(p.name for p in (folder/'graficos').glob('*.png'))}
