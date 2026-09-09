import copy
import os
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
from .datos import FEATURES, secuencias


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
        print('Entrenando red PyTorch...', flush=True)
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.001)
        lossfn = torch.nn.BCEWithLogitsLoss()
        xt, yt = torch.from_numpy(z[tr]), torch.from_numpy(y[tr].astype('float32')).view(-1, 1)
        xv, yv = torch.from_numpy(z[va]), torch.from_numpy(y[va].astype('float32')).view(-1, 1)
        best, stale, state, history = float('inf'), 0, None, []
        for epoch in range(epochs):
            net.train()
            total = 0.
            for start in range(0, len(xt), 64):
                optimizer.zero_grad()
                loss = lossfn(net(xt[start:start+64]), yt[start:start+64])
                loss.backward()
                optimizer.step()
                total += loss.item()*len(xt[start:start+64])
            net.eval()
            with torch.no_grad():
                vl = lossfn(net(xv), yv).item()
            history.append({'epoch': epoch+1, 'loss': total/len(xt), 'val_loss': vl})
            if vl < best-1e-5:
                best, stale, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if stale >= 5:
                break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            probabilities['PyTorch_MLP'] = torch.sigmoid(net(torch.from_numpy(z[te]))).numpy().ravel()
            validation['PyTorch_MLP'] = evaluar(y[va], torch.sigmoid(net(xv)).numpy().ravel())
        torch.save(net.state_dict(), modeldir / 'pytorch_mlp.pt')
        histories['PyTorch_MLP'] = history

        print('Entrenando LSTM TensorFlow/Keras...', flush=True)
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        train_seq = tr[tr >= window-1]
        xs, vs, ts = (secuencias(z, idx, window) for idx in [train_seq, va, te])
        lstm = tf.keras.Sequential([tf.keras.Input(shape=(window, len(FEATURES))),
                                   tf.keras.layers.LSTM(16), tf.keras.layers.Dense(8, activation='relu'),
                                   tf.keras.layers.Dense(1, activation='sigmoid')])
        lstm.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='binary_crossentropy')
        hist = lstm.fit(xs, y[train_seq], validation_data=(vs, y[va]), epochs=epochs,
                        batch_size=64, shuffle=False, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                                    restore_best_weights=True)])
        probabilities['Keras_LSTM'] = lstm.predict(ts, verbose=0).ravel()
        validation['Keras_LSTM'] = evaluar(y[va], lstm.predict(vs, verbose=0).ravel())
        lstm.save(modeldir / 'keras_lstm.keras')
        histories['Keras_LSTM'] = [{'epoch': i+1, 'loss': float(l), 'val_loss': float(v)}
                                 for i, (l, v) in enumerate(zip(hist.history['loss'], hist.history['val_loss']))]
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
