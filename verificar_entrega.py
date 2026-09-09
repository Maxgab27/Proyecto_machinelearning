"""Verifica artefactos reales sin repetir entrenamiento."""
import os
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('KERAS_HOME', str(ROOT / 'tmp' / 'keras'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))


def main():
    from mercado.datos import preparar, particiones, secuencias, FEATURES
    from mercado.modelos import evaluar
    from mercado.informe import escribir
    out = ROOT/'resultados'
    summary = json.loads((out/'resumen.json').read_text(encoding='utf-8'))
    assert (out/'COMPLETADO.json').is_file(), 'La ejecución no está completa.'
    df, _ = preparar(ROOT/'datos'/'arch_financiero.csv')
    parts = particiones(df)
    te = parts['test']
    x = df[FEATURES].to_numpy(dtype='float32')
    stored = pd.read_csv(out/'predicciones.csv')
    assert len(stored) == len(te)
    assert summary['datos']['sha256_csv'] == hashlib.sha256((ROOT/'datos'/'arch_financiero.csv').read_bytes()).hexdigest()
    probs = {}
    for name, filename in [('Logistica', 'logistica.joblib'), ('Base_mayoritaria', 'base.joblib')]:
        probs[name] = joblib.load(out/'modelos'/filename).predict_proba(x[te])[:, 1]
    scaler = joblib.load(out/'modelos'/'escalador_redes.joblib')
    np.testing.assert_allclose(scaler.mean_, x[parts['train']].mean(axis=0, dtype='float64'), rtol=1e-6)
    z = scaler.transform(x).astype('float32')
    if summary['configuracion']['modo'] == 'completo':
        import torch
        torch.set_num_threads(2)
        net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        net.load_state_dict(torch.load(out/'modelos'/'pytorch_mlp.pt', weights_only=True))
        net.eval()
        with torch.no_grad():
            probs['PyTorch_MLP'] = torch.sigmoid(net(torch.from_numpy(z[te]))).numpy().ravel()
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        model = tf.keras.models.load_model(out/'modelos'/'keras_lstm.keras')
        probs['Keras_LSTM'] = model.predict(secuencias(z, te, summary['configuracion']['ventana']), verbose=0).ravel()
    for name, probability in probs.items():
        np.testing.assert_allclose(probability, stored[f'{name}_probabilidad'], atol=1e-6)
        np.testing.assert_array_equal((probability>=.5).astype(int), stored[f'{name}_clase'])
        score = evaluar(df.Objetivo.iloc[te], probability)
        assert score['matriz_confusion'] == summary['metricas_test'][name]['matriz_confusion']
    # Regenerar solo el informe para incorporar cambios editoriales sin reentrenar.
    escribir(summary, out)
    verification = {'estado':'correcto', 'modelos_recargados':list(probs),
                    'predicciones_verificadas_por_modelo':len(te),
                    'tolerancia_probabilidades':1e-6,
                    'csv_original_verificado': True, 'escalador_solo_entrenamiento':True}
    (out/'verificacion.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
