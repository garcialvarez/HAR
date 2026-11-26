"""
training.py
-----------
Entrenamiento y evaluación de modelos Keras, comunes a los notebooks 04–07.

Los modelos (la arquitectura) se definen dentro de cada notebook para que se vean; aquí solo está el
bucle de entrenamiento, la preparación de entradas con partición por sujeto y la evaluación.
"""

from __future__ import annotations

import time

import numpy as np

from evaluation import evaluate_and_save, keras_model_size_kb, measure_latency
from preprocessing import ChannelStandardizer, make_channels, subject_train_val_split


def prepare_time_inputs(X_train_raw, X_test_raw, subjects_train, variant: str = "raw", split_seed: int = 42):
    """
    Prepara las entradas (N, T, C) para un modelo en el dominio del tiempo.

    1. Construye los canales según `variant` (ver preprocessing.make_channels).
    2. Separa entrenamiento/validación POR SUJETO dentro del train oficial.
    3. Ajusta el estandarizador solo con las ventanas de entrenamiento.

    Devuelve un diccionario con Xtr, Xva, Xte (float32) y las máscaras tr/va sobre el train oficial.
    """
    A = make_channels(X_train_raw, variant)
    T = make_channels(X_test_raw, variant)
    tr, va = subject_train_val_split(subjects_train, seed=split_seed)
    sc = ChannelStandardizer().fit(A[tr])
    return {"Xtr": sc.transform(A[tr]), "Xva": sc.transform(A[va]), "Xte": sc.transform(T),
            "tr": tr, "va": va}


def prepare_stft_inputs(X_train_raw, X_test_raw, subjects_train, nperseg: int = 32,
                        variant: str = "raw", split_seed: int = 42):
    """
    Igual que `prepare_time_inputs`, pero convierte cada canal a espectrograma (STFT).

    Devuelve Xtr, Xva, Xte con forma (N, F, T', C), estandarizados por canal con estadísticos del
    entrenamiento, más las máscaras tr/va sobre el train oficial.
    """
    from preprocessing import compute_stft

    A = compute_stft(make_channels(X_train_raw, variant), nperseg=nperseg)
    T = compute_stft(make_channels(X_test_raw, variant), nperseg=nperseg)
    tr, va = subject_train_val_split(subjects_train, seed=split_seed)
    sc = ChannelStandardizer().fit(A[tr])
    return {"Xtr": sc.transform(A[tr]), "Xva": sc.transform(A[va]), "Xte": sc.transform(T),
            "tr": tr, "va": va}


def fit_keras(model, Xtr, ytr, Xva, yva, epochs: int = 40, batch_size: int = 64,
              patience: int = 8, lr: float = 1e-3, verbose: int = 0):
    """
    Compila y entrena con Adam, EarlyStopping sobre val_loss (restaura los mejores pesos) y
    ReduceLROnPlateau. Devuelve (history, segundos de entrenamiento).
    """
    from tensorflow import keras

    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2)),
    ]
    t0 = time.perf_counter()
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=verbose)
    return history, time.perf_counter() - t0


def evaluate_keras(model_id: str, family: str, model, Xte, yte, class_names, train_time_s: float,
                   notes: str = "", save: bool = True) -> dict:
    """
    Evalúa en el test: métricas, latencia con lote de 1, parámetros y tamaño del modelo.

    La latencia se mide sobre el grafo compilado (`tf.function`), que es lo que correría en un
    despliegue; la llamada "eager" de Keras añade ~20 ms de sobrecosto fijo por llamada.
    """
    import tensorflow as tf

    spec = tf.TensorSpec((None,) + tuple(Xte.shape[1:]), tf.float32)
    infer = tf.function(lambda x: model(x, training=False), input_signature=[spec])

    def predict(x):
        return np.argmax(infer(tf.constant(x, tf.float32)).numpy(), axis=1)

    y_pred = model.predict(Xte, verbose=0).argmax(axis=1)
    latency = measure_latency(predict, Xte)
    return evaluate_and_save(model_id, family, yte, y_pred, class_names, train_time_s=train_time_s,
                             latency=latency, n_params=model.count_params(),
                             model_size_kb=keras_model_size_kb(model), notes=notes, save=save)
