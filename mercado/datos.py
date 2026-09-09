from pathlib import Path
import csv
import numpy as np
import pandas as pd

FEATURES = ['open', 'high', 'low', 'close', 'Log_Volumen', 'Retorno',
            'Distancia_MA7', 'Distancia_MA30', 'Volatilidad_7']


def preparar(ruta: Path, max_rows=None):
    if max_rows is not None and max_rows < 1:
        raise ValueError('El límite de filas debe ser positivo.')
    # Rechazar cabeceras patológicas antes de crear miles de columnas en Pandas.
    with ruta.open(encoding='utf-8-sig', newline='') as source:
        header = source.readline(8193)
    if len(header) > 8192:
        raise ValueError('La cabecera del CSV supera los 8 KB permitidos.')
    columns = next(csv.reader([header]), [])
    if len(columns) > 64 or len(columns) != len(set(columns)):
        raise ValueError('El CSV contiene demasiadas columnas o nombres repetidos.')
    if 'Date' in columns and 'Unnamed: 0' in columns:
        raise ValueError('Utiliza una sola columna de fecha: Date o Unnamed: 0.')
    df = pd.read_csv(ruta, nrows=max_rows+1 if max_rows is not None else None).rename(columns={'Unnamed: 0': 'Date'})
    if max_rows is not None and len(df) > max_rows:
        raise ValueError(f'El límite es {max_rows:,} filas de entrada por CSV.')
    required = {'Date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
    if required - set(df.columns):
        raise ValueError(f'Faltan columnas: {sorted(required - set(df.columns))}')
    original = len(df)
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')
    df = df.drop_duplicates().sort_values('Date').reset_index(drop=True)
    if df['ticker'].nunique() != 1 or df['ticker'].isna().any():
        raise ValueError('Esta versión requiere exactamente un ticker por CSV.')
    if df['Date'].isna().any() or df['Date'].duplicated().any():
        raise ValueError('Hay fechas vacías o registros conflictivos para la misma fecha.')
    numeric = ['open', 'high', 'low', 'close', 'volume']
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError('El CSV contiene datos faltantes o infinitos; corregir la fuente.')
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Precios no positivos o volumen negativo.')
    if ((df.high < df[['open', 'close', 'low']].max(axis=1)) |
            (df.low > df[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('Los máximos y mínimos OHLC son inconsistentes.')
    df['Retorno'] = df.close.pct_change(fill_method=None)
    df['Log_Volumen'] = np.log1p(df.volume.to_numpy(dtype=np.float64))
    for window in (7, 30):
        df[f'Media_Movil_{window}'] = df.close.rolling(window).mean()
        df[f'Distancia_MA{window}'] = df.close / df[f'Media_Movil_{window}'] - 1
    df['Volatilidad_7'] = df.Retorno.rolling(7).std()
    df['Close_Siguiente'] = df.close.shift(-1)
    df['Fecha_Objetivo'] = df.Date.shift(-1)
    # El último registro no tiene etiqueta y nunca se convierte en una bajada ficticia.
    df = df.dropna(subset=FEATURES + ['Close_Siguiente', 'Fecha_Objetivo']).copy()
    df['Objetivo'] = (df.Close_Siguiente > df.close).astype('int8')
    df[FEATURES] = df[FEATURES].astype('float32')
    df = df.reset_index(drop=True)
    if len(df) < 150:
        raise ValueError('Se requieren al menos 150 observaciones válidas.')
    audit = {'filas_originales': original, 'filas_utiles': len(df),
             'filas_descartadas': original-len(df), 'ticker': str(df.ticker.iloc[0]),
             'inicio': str(df.Date.iloc[0].date()), 'fin': str(df.Date.iloc[-1].date()),
             'memoria_features_bytes': int(df[FEATURES].memory_usage(index=False).sum())}
    return df, audit


def particiones(df):
    """60/20/20; purgar una etiqueta en cada frontera de selección."""
    a, b = int(len(df)*.6), int(len(df)*.8)
    parts = {'train': np.arange(0, a-1), 'val': np.arange(a, b-1),
             'test': np.arange(b, len(df))}
    for left, right in [('train', 'val'), ('val', 'test')]:
        if df.Fecha_Objetivo.iloc[parts[left]].max() >= df.Date.iloc[parts[right]].min():
            raise ValueError('Las etiquetas cruzan las fronteras temporales.')
    if df.Objetivo.iloc[parts['train']].nunique() < 2:
        raise ValueError('El entrenamiento necesita ambas clases.')
    return parts


def secuencias(x, indices, ventana=20):
    if ventana < 2 or len(indices) == 0 or min(indices) < ventana-1:
        raise ValueError('Índices o ventana insuficientes para formar secuencias.')
    return np.stack([x[i-ventana+1:i+1] for i in indices]).astype('float32')
