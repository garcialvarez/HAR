"""
preprocessing.py
----------------
Preprocesado de las señales inerciales del UCI HAR Dataset.

Convenciones
------------
- Las ventanas crudas tienen forma (N, T, C) = (N, 128, 9), con los canales en el orden de
  `data_loader.INERTIAL_SIGNALS`: body_acc xyz (0-2), body_gyro xyz (3-5), total_acc xyz (6-8).
- body_acc y total_acc están en unidades de g; el giroscopio, en rad/s. Fs = 50 Hz.
- Todo lo que se "ajusta" (media/desviación) se ajusta SOLO con datos de entrenamiento.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import GroupShuffleSplit

from data_loader import ACTIVITY_NAMES

FS = 50.0           # Hz
G = 9.80665         # m/s^2 por g
CLASS_NAMES = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES)]  # orden de índices 0..5


def encode_labels(y_names) -> np.ndarray:
    """Convierte nombres de actividad a índices 0..5 (orden de CLASS_NAMES)."""
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    return np.asarray([mapping[n] for n in y_names], dtype=np.int64)


def subject_train_val_split(subjects, val_frac: float = 0.2, seed: int = 42):
    """
    Separa entrenamiento/validación por SUJETO (ningún sujeto en ambos conjuntos).

    Devuelve dos máscaras booleanas (train_mask, val_mask) sobre las ventanas.
    """
    subjects = np.asarray(subjects)
    idx = np.arange(len(subjects))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(idx, groups=subjects))
    assert not set(subjects[tr]) & set(subjects[va]), "Fuga: sujetos repetidos entre train y val"
    train_mask = np.zeros(len(subjects), dtype=bool)
    val_mask = np.zeros(len(subjects), dtype=bool)
    train_mask[tr] = True
    val_mask[va] = True
    return train_mask, val_mask


class ChannelStandardizer:
    """Estandarización por canal (último eje). Media/desviación se calculan sobre todos los demás ejes."""

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        axes = tuple(range(X.ndim - 1))
        self.mean_ = X.mean(axis=axes, keepdims=True)
        self.std_ = X.std(axis=axes, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def add_magnitudes(X: np.ndarray) -> np.ndarray:
    """
    Agrega la magnitud de body_acc, body_gyro y total_acc (invariantes a la orientación del
    teléfono). (N, T, 9) -> (N, T, 12).
    """
    mags = [np.linalg.norm(X[..., s:s + 3], axis=-1, keepdims=True) for s in (0, 3, 6)]
    return np.concatenate([X] + mags, axis=-1)


def integrate_velocity(body_acc: np.ndarray, fs: float = FS, highpass_hz: float | None = 0.3) -> np.ndarray:
    """
    Velocidad a partir de la aceleración corporal (N, T, 3) en g -> (N, T, 3) en m/s.

    Pasos: g -> m/s^2, quitar la media de cada ventana (elimina el offset del sensor) e integrar
    con la regla del trapecio. Si `highpass_hz` no es None, se aplica además un pasa-altos
    Butterworth (orden 3, fase cero) a la velocidad para frenar la deriva. Cada ventana se integra
    por separado, así que la velocidad es relativa al inicio de la ventana.
    """
    a = body_acc * G
    a = a - a.mean(axis=1, keepdims=True)
    v = cumulative_trapezoid(a, dx=1.0 / fs, axis=1, initial=0.0)
    if highpass_hz is not None:
        b, a_coef = signal.butter(3, highpass_hz, btype="highpass", fs=fs)
        v = signal.filtfilt(b, a_coef, v, axis=1)
    return v.astype(np.float32)


def make_channels(X: np.ndarray, variant: str = "raw") -> np.ndarray:
    """
    Construye el conjunto de canales de entrada para los modelos en el dominio del tiempo.

    - "raw":     los 9 canales originales.
    - "mag":     9 + 3 magnitudes = 12 canales.
    - "mag_vel": 12 + 3 de velocidad integrada y corregida (body_acc) = 15 canales.
    """
    if variant == "raw":
        return X
    M = add_magnitudes(X)
    if variant == "mag":
        return M
    if variant == "mag_vel":
        return np.concatenate([M, integrate_velocity(X[..., 0:3])], axis=-1)
    raise ValueError(f"variante desconocida: {variant!r}")


def compute_stft(X: np.ndarray, fs: float = FS, nperseg: int = 32, noverlap: int | None = None) -> np.ndarray:
    """
    Espectrograma log(1+|STFT|) por canal. (N, T, C) -> (N, F, T', C).

    Con nperseg=32 y noverlap=28 -> F=17, T'=25. Con nperseg=64 y noverlap=56 -> F=33, T'=9.
    """
    if noverlap is None:
        noverlap = nperseg * 7 // 8
    _, _, Z = signal.stft(X, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1,
                          boundary=None, padded=False)
    # scipy devuelve (N, F, C, T'); lo llevamos a (N, F, T', C)
    S = np.log1p(np.abs(Z)).transpose(0, 1, 3, 2)
    return S.astype(np.float32)
