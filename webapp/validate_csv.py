"""Valida en un hijo para no retener Pandas/NumPy en el servidor HTTP."""
import json
import re
import sys
from pathlib import Path


def main():
    from mercado.datos import preparar, particiones
    path, result, max_rows = sys.argv[1:]
    try:
        df, audit = preparar(Path(path), max_rows=int(max_rows))
        particiones(df)
        if not re.fullmatch(r'[A-Za-z0-9.^=_-]{1,24}', audit['ticker']):
            raise ValueError('El ticker debe ser un identificador bursátil de hasta 24 caracteres.')
        payload = {'audit': audit}
    except (ValueError, OSError) as exc:
        payload = {'error': str(exc)}
    Path(result).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
