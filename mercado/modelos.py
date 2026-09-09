import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, log_loss)
from .datos import FEATURES


def evaluar(y, probability):
    probability = np.asarray(probability).reshape(-1)
    if len(probability) != len(y) or not np.isfinite(probability).all():
        raise ValueError('Predicciones inválidas.')
    pred = (probability >= .5).astype(int)
    return {'exactitud': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            'log_loss': float(log_loss(y, probability, labels=[0, 1])),
            'matriz_confusion': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def entrenar(df, parts, out, epochs=30, window=20, seed=42, mode='completo'):
    modeldir = out / 'modelos'
    modeldir.mkdir(parents=True, exist_ok=True)
    x = df[FEATURES].to_numpy(dtype='float32')
    y = df.Objetivo.to_numpy()
    tr, va, te = (parts[k] for k in ['train', 'val', 'test'])
    probabilities, metrics, histories = {}, {}, {}
    cv = []
    # Elegir C exclusivamente mediante validación temporal dentro de entrenamiento.
    for c in [.01, .1, 1., 10.]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
        scores = cross_val_score(pipe, x[tr], y[tr],
                                 cv=TimeSeriesSplit(n_splits=3, gap=1), scoring='neg_log_loss')
        cv.append({'C': c, 'log_loss_cv': float(-scores.mean())})
    selected = min(cv, key=lambda row: row['log_loss_cv'])['C']
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=selected, max_iter=2000, random_state=seed))
    logistic.fit(x[tr], y[tr])
    probabilities['Logistica'] = logistic.predict_proba(x[te])[:, 1]
    joblib.dump(logistic, modeldir / 'logistica.joblib')
    baseline = DummyClassifier(strategy='most_frequent').fit(x[tr], y[tr])
    probabilities['Base_mayoritaria'] = baseline.predict_proba(x[te])[:, 1]
    joblib.dump(baseline, modeldir / 'base.joblib')
    pd.DataFrame(cv).to_csv(out / 'validacion_cruzada.csv', index=False)
    validation = {'Logistica': evaluar(y[va], logistic.predict_proba(x[va])[:, 1])}
    scaler = StandardScaler().fit(x[tr])
    z = scaler.transform(x).astype('float32')
    joblib.dump(scaler, modeldir / 'escalador_redes.joblib')

    if mode == 'completo':
        # Al salir cada hijo, el SO libera su runtime completo antes del siguiente.
        with tempfile.TemporaryDirectory(prefix='.redes-', dir=out) as tmp:
            inputs = Path(tmp)/'entrada.npz'
            np.savez(inputs, z=z, y=y, tr=tr, va=va, te=te)
            for engine in ('pytorch', 'keras'):
                print(f'Entrenando {engine} en proceso aislado...', flush=True)
                output = Path(tmp)/f'{engine}.json'
                subprocess.run([sys.executable, '-X', 'utf8', '-m', 'mercado.redes_worker',
                                engine, str(inputs.resolve()), str(output.resolve()),
                                str(modeldir.resolve()), str(epochs), str(window), str(seed)],
                               cwd=Path(__file__).resolve().parents[1], check=True)
                result = json.loads(output.read_text(encoding='utf-8'))
                probabilities.update({k:np.asarray(v) for k,v in result['probabilities'].items()})
                validation.update({k:evaluar(y[va], v) for k,v in result['validation'].items()})
                histories.update(result['histories'])
    prediction = df.iloc[te][['Date', 'Fecha_Objetivo', 'close', 'Close_Siguiente', 'Objetivo']].copy()
    for name, probability in probabilities.items():
        metrics[name] = evaluar(y[te], probability)
        prediction[f'{name}_probabilidad'] = probability
        prediction[f'{name}_clase'] = (probability >= .5).astype(int)
    prediction.to_csv(out / 'predicciones.csv', index=False)
    pd.DataFrame([{ 'modelo': name, **{k:v for k,v in score.items() if k!='matriz_confusion'}}
                  for name, score in metrics.items()]).to_csv(out / 'metricas.csv', index=False)
    for name, history in histories.items():
        pd.DataFrame(history).to_csv(out / f'historial_{name}.csv', index=False)
    return metrics, validation, histories, prediction, selected
