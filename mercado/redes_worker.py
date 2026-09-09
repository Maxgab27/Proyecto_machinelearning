"""Motores aislados: cada proceso carga una sola biblioteca de redes."""
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
from .datos import FEATURES


def pytorch(z, y, tr, va, te, epochs, window, seed, modeldir):
    probabilities, validation, histories = {}, {}, {}
    import torch
    torch.set_num_threads(int(os.environ.get('ML_NUM_THREADS', '1')))
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
        validation['PyTorch_MLP'] = torch.sigmoid(net(xv)).numpy().ravel()
    torch.save(net.state_dict(), modeldir / 'pytorch_mlp.pt')
    histories['PyTorch_MLP'] = history

    return probabilities, validation, histories


def keras(z, y, tr, va, te, epochs, window, seed, modeldir):
    probabilities, validation, histories = {}, {}, {}
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('ML_NUM_THREADS', '1')))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    train_seq = tr[tr >= window-1]
    # Construir solamente las ventanas del lote actual, no N ventanas en RAM.
    features = tf.constant(z)
    offsets = tf.range(1-window, 1, dtype=tf.int64)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1

    def dataset(indices):
        ds = tf.data.Dataset.from_tensor_slices((indices.astype('int64'), y[indices]))
        ds = ds.batch(64).map(lambda ids, labels: (
            tf.gather(features, ids[:, None]+offsets[None, :]), labels),
            num_parallel_calls=1)
        return ds.with_options(options).prefetch(1)

    xs, vs, ts = (dataset(idx) for idx in [train_seq, va, te])
    lstm = tf.keras.Sequential([tf.keras.Input(shape=(window, len(FEATURES))),
                               tf.keras.layers.LSTM(16), tf.keras.layers.Dense(8, activation='relu'),
                               tf.keras.layers.Dense(1, activation='sigmoid')])
    lstm.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='binary_crossentropy')
    hist = lstm.fit(xs, validation_data=vs, epochs=epochs,
                    shuffle=False, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                                restore_best_weights=True)])
    probabilities['Keras_LSTM'] = lstm.predict(ts, verbose=0).ravel()
    validation['Keras_LSTM'] = lstm.predict(vs, verbose=0).ravel()
    lstm.save(modeldir / 'keras_lstm.keras')
    histories['Keras_LSTM'] = [{'epoch': i+1, 'loss': float(l), 'val_loss': float(v)}
                             for i, (l, v) in enumerate(zip(hist.history['loss'], hist.history['val_loss']))]
    return probabilities, validation, histories


def main():
    engine, input_path, output_path, model_path, epochs, window, seed = sys.argv[1:]
    with np.load(input_path, allow_pickle=False) as data:
        args = [data[k] for k in ('z', 'y', 'tr', 'va', 'te')]
    fn = {'pytorch': pytorch, 'keras': keras}[engine]
    probabilities, validation, histories = fn(*args, int(epochs), int(window), int(seed), Path(model_path))
    Path(output_path).write_text(json.dumps({
        'probabilities': {k:v.tolist() for k,v in probabilities.items()},
        'validation': {k:v.tolist() for k,v in validation.items()},
        'histories': histories,
    }), encoding='utf-8')


if __name__ == '__main__':
    main()
