"""
data_loader.py
----------------
Utilidades para cargar el dataset "Human Activity Recognition Using
Smartphones" (UCI HAR Dataset) a partir del .zip original o de una carpeta
ya descomprimida.

El dataset debe estar disponible en `data/raw/UCI_HAR_Dataset.zip`
(ver data/README.md; `download_dataset()` lo descarga). Si aún no ha sido descomprimido, este módulo
lo descomprime automáticamente en `data/raw/UCI_HAR_Dataset/`.
"""

from __future__ import annotations

import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "UCI_HAR_Dataset.zip"
EXTRACT_DIR = RAW_DIR / "UCI_HAR_Dataset"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

INERTIAL_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def ensure_extracted() -> Path:
    """Descomprime el dataset si aún no ha sido extraído. Devuelve la ruta base."""
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        return EXTRACT_DIR

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {ZIP_PATH}. Coloca el archivo UCI_HAR_Dataset.zip "
            "en data/raw/ (ver README.md, sección Dataset)."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # Ignoramos metadatos de macOS si estuvieran presentes en el zip.
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        zf.extractall(RAW_DIR, members=members)

    # El .zip original contiene la carpeta "UCI HAR Dataset" (con espacio);
    # la normalizamos a "UCI_HAR_Dataset" para evitar problemas de rutas.
    spaced_dir = RAW_DIR / "UCI HAR Dataset"

    # La distribución actual de UCI empaqueta el dataset como un .zip anidado
    # (RAW_DIR/"UCI HAR Dataset.zip" en vez de la carpeta directamente).
    inner_zip = RAW_DIR / "UCI HAR Dataset.zip"
    if not spaced_dir.exists() and inner_zip.exists():
        with zipfile.ZipFile(inner_zip) as zf:
            members = [m for m in zf.namelist() if "__MACOSX" not in m]
            zf.extractall(RAW_DIR, members=members)

    if spaced_dir.exists() and not EXTRACT_DIR.exists():
        spaced_dir.rename(EXTRACT_DIR)

    return EXTRACT_DIR


def download_dataset(retries: int = 3) -> Path:
    """
    Descarga el UCI HAR Dataset a `data/raw/UCI_HAR_Dataset.zip` (si no existe)
    y lo descomprime. Devuelve la ruta de la carpeta extraída.

    Si la descarga falla tras `retries` intentos, indica cómo hacerla a mano.
    """
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        return EXTRACT_DIR

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                print(f"Descargando dataset (intento {attempt}/{retries})...")
                req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(ZIP_PATH, "wb") as f:
                    f.write(resp.read())
                break
            except Exception as err:  # red caída, timeout, 5xx...
                last_err = err
                ZIP_PATH.unlink(missing_ok=True)
                time.sleep(2 * attempt)
        else:
            raise RuntimeError(
                f"No se pudo descargar el dataset ({last_err}). Descárgalo a mano desde "
                f"{DATASET_URL} y guárdalo como {ZIP_PATH} (ver data/README.md)."
            )
        print(f"Descargado: {ZIP_PATH.stat().st_size / 1e6:.1f} MB")

    return ensure_extracted()


def _load_split(base: Path, split: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga las 561 features, etiquetas y sujetos de un split ('train' o 'test')."""
    features = pd.read_csv(base / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    # Los nombres de features no son únicos (p. ej. "fBodyAcc-bandsEnergy()-1,8"
    # se repite en distintos ejes); se desambiguan añadiendo el índice.
    feature_names = [f"{i}_{n}" for i, n in zip(features["idx"], features["name"])]

    X = pd.read_csv(base / split / f"X_{split}.txt", sep=r"\s+", header=None, names=feature_names)
    y = pd.read_csv(base / split / f"y_{split}.txt", header=None, names=["activity_id"])["activity_id"]
    subjects = pd.read_csv(base / split / f"subject_{split}.txt", header=None, names=["subject_id"])["subject_id"]

    return X, y, subjects


def load_features(split: str = "train") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga el vector de 561 características ya extraídas por los autores del dataset.

    Parameters
    ----------
    split : "train" o "test"

    Returns
    -------
    X : DataFrame (n_samples, 561)
    y : Series con el nombre de la actividad (string)
    subjects : Series con el id de sujeto (1-30)
    """
    base = ensure_extracted()
    X, y_id, subjects = _load_split(base, split)
    y = y_id.map(ACTIVITY_NAMES)
    return X, y, subjects


def load_raw_signals(split: str = "train") -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Carga las señales inerciales crudas (9 canales x 128 pasos por ventana),
    útiles para modelos de deep learning (CNN/RNN) que aprenden features
    automáticamente en lugar de usar el vector de 561 features pre-calculadas.

    Returns
    -------
    X : ndarray de forma (n_samples, 128, 9)
    y : Series con el nombre de la actividad (string)
    subjects : Series con el id de sujeto (1-30)
    """
    base = ensure_extracted()
    signal_dir = base / split / "Inertial Signals"

    channels = []
    for sig in INERTIAL_SIGNALS:
        arr = np.loadtxt(signal_dir / f"{sig}_{split}.txt")
        channels.append(arr)
    X = np.stack(channels, axis=-1)  # (n_samples, 128, 9)

    y_id = pd.read_csv(base / split / f"y_{split}.txt", header=None, names=["activity_id"])["activity_id"]
    y = y_id.map(ACTIVITY_NAMES)
    subjects = pd.read_csv(base / split / f"subject_{split}.txt", header=None, names=["subject_id"])["subject_id"]

    return X, y, subjects


def load_all_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Concatena train + test (10,299 ventanas, 30 sujetos). Para validación cruzada por sujeto."""
    Xa, ya, sa = load_features("train")
    Xb, yb, sb = load_features("test")
    return (pd.concat([Xa, Xb], ignore_index=True),
            pd.concat([ya, yb], ignore_index=True),
            pd.concat([sa, sb], ignore_index=True))


def load_all_raw() -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Señales crudas de train + test concatenadas: X (10299, 128, 9)."""
    Xa, ya, sa = load_raw_signals("train")
    Xb, yb, sb = load_raw_signals("test")
    return (np.concatenate([Xa, Xb], axis=0),
            pd.concat([ya, yb], ignore_index=True),
            pd.concat([sa, sb], ignore_index=True))


if __name__ == "__main__":
    X_train, y_train, subj_train = load_features("train")
    X_test, y_test, subj_test = load_features("test")
    print(f"Train: {X_train.shape}, clases: {y_train.value_counts().to_dict()}")
    print(f"Test:  {X_test.shape}, clases: {y_test.value_counts().to_dict()}")
    print(f"Sujetos train: {sorted(subj_train.unique())}")
    print(f"Sujetos test:  {sorted(subj_test.unique())}")
