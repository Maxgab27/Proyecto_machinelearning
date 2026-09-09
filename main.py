"""Ejecutar desde cualquier directorio: python /ruta/al/proyecto/main.py."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
os.environ.setdefault('KERAS_HOME', str(ROOT / 'tmp' / 'keras'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def main():
    parser = argparse.ArgumentParser(description='Trabajo final: tendencias financieras y NLP')
    parser.add_argument('--csv', type=Path, default=ROOT / 'datos' / 'arch_financiero.csv')
    parser.add_argument('--reportes', type=Path, default=ROOT / 'datos' / 'reportes')
    parser.add_argument('--salida', type=Path, default=ROOT / 'resultados')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ventana', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-filas', type=int, default=None,
                        help='Límite de filas de entrada para el servicio web')
    parser.add_argument('--modo', choices=['completo', 'base'], default='completo',
                        help='base sirve para diagnóstico; no cubre las redes del entregable')
    args = parser.parse_args()
    if args.epochs < 1 or args.ventana < 2:
        parser.error('epochs debe ser positivo y ventana debe ser al menos 2.')
    from mercado.datos import preparar, particiones, FEATURES
    from mercado.modelos import entrenar
    from mercado.nlp import analizar
    from mercado.informe import escribir
    import numpy as np
    np.random.seed(args.seed)
    out = args.salida.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'COMPLETADO.json').unlink(missing_ok=True)
    print('Validando datos y creando indicadores...', flush=True)
    df, audit = preparar(args.csv, max_rows=args.max_filas)
    parts = particiones(df)
    if args.ventana > len(parts['train']) // 2:
        parser.error('La ventana es demasiado grande para el conjunto de entrenamiento.')
    with args.csv.open('rb') as source:
        audit['sha256_csv'] = hashlib.file_digest(source, 'sha256').hexdigest()
    audit['particiones'] = {name: {'filas': len(idx), 'inicio': str(df.Date.iloc[idx[0]].date()),
                                  'fin': str(df.Date.iloc[idx[-1]].date()),
                                  'ultima_fecha_objetivo': str(df.Fecha_Objetivo.iloc[idx[-1]].date())}
                            for name, idx in parts.items()}
    df.to_csv(out / 'datos_preparados.csv', index=False)
    print('Analizando reportes financieros...', flush=True)
    nlp, terms = analizar(args.reportes, out)
    print('Validando y entrenando modelos...', flush=True)
    metrics, validation, histories, prediction, selected = entrenar(
        df, parts, out, epochs=args.epochs, window=args.ventana, seed=args.seed, mode=args.modo)
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'nltk', 'torch', 'tensorflow', 'keras', 'matplotlib', 'seaborn', 'pypdf']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if args.modo == 'completo' or name not in {'torch', 'tensorflow', 'keras'}:
                raise
            versions[name] = 'No instalado (modo base)'
    result = {'estado': 'completado', 'fecha_utc': datetime.now(timezone.utc).isoformat(),
              'configuracion': {'modo': args.modo, 'epochs_max': args.epochs, 'ventana': args.ventana,
                                'seed': args.seed, 'C_logistica': selected, 'features': FEATURES},
              'datos': audit, 'metricas_test': metrics, 'metricas_validacion': validation,
              'nlp': nlp, 'versiones': versions}
    (out / 'resumen.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    from mercado.graficos import generar
    generar(df, metrics, histories, prediction, terms, out / 'graficos')
    escribir(result, out)
    (out / 'COMPLETADO.json').write_text(json.dumps({'fecha_utc': result['fecha_utc'], 'modo': args.modo}), encoding='utf-8')
    print('\nResultados de prueba (exactitud / F1):')
    for name, score in metrics.items():
        print(f"  {name}: {score['exactitud']:.2%} / {score['f1']:.4f}")
    print(f'\nInforme generado: {out / "INFORME_TECNICO.md"}')


if __name__ == '__main__':
    main()
