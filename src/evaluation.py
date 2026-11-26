"""
evaluation.py
-------------
Métricas, latencia, tamaño de modelo y gráficas comunes a todos los notebooks.

`evaluate_and_save` calcula las métricas de ML y de negocio de la propuesta y las guarda en
`results/<model_id>.json` (carpeta no versionada: se usa solo para armar el informe en local).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def measure_latency(predict_fn, X: np.ndarray, n: int = 100, rounds: int = 3) -> dict:
    """
    Latencia de inferencia en ms por ventana.

    predict_fn: función que recibe un lote (k, ...) y devuelve predicciones.
    - single_ms: mediana con lote de 1 (escenario on-device). Se repite en `rounds` rondas y se toma la
      mejor, para que una interferencia puntual del sistema no distorsione la cifra.
    - batch_ms: costo por ventana procesando todo el conjunto de una vez.
    """
    n = min(n, len(X))
    predict_fn(X[:1])  # calentamiento (compilación / caches)
    medianas = []
    for _ in range(rounds):
        single = []
        for i in range(n):
            t0 = time.perf_counter()
            predict_fn(X[i:i + 1])
            single.append((time.perf_counter() - t0) * 1000)
        medianas.append(np.median(single))
    t0 = time.perf_counter()
    predict_fn(X)
    batch_ms = (time.perf_counter() - t0) * 1000 / len(X)
    return {"single_ms": float(min(medianas)), "batch_ms": float(batch_ms)}


def keras_model_size_kb(model) -> float:
    """
    Tamaño en KB de los pesos del modelo en float32 (4 bytes por valor), que es lo que habría que
    desplegar. No se usa `model.save`/`save_weights`: en Keras 3 guardan también el estado del
    optimizador (Adam), lo que triplica el archivo y no se necesita para inferir.
    """
    return sum(int(np.prod(w.shape)) for w in model.weights) * 4 / 1024


def evaluate_and_save(model_id: str, family: str, y_true, y_pred, class_names, *,
                      train_time_s=None, latency=None, n_params=None, model_size_kb=None,
                      notes: str = "", save: bool = True) -> dict:
    """
    Calcula las métricas, las guarda en results/<model_id>.json y devuelve el diccionario.

    y_true / y_pred: índices 0..K-1. `latency`: salida de measure_latency (o None).
    La tasa de falsos negativos por clase es 1 - recall (relevante en clases de riesgo).
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    res = {
        "model_id": model_id,
        "family": family,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {k: rep[k] for k in class_names},
        "false_negative_rate": {c: float(1 - r) for c, r in zip(class_names, rec)},
        "confusion_matrix": cm.tolist(),
        "train_time_s": None if train_time_s is None else float(train_time_s),
        "latency_single_ms": None if latency is None else latency["single_ms"],
        "latency_batch_ms": None if latency is None else latency["batch_ms"],
        "n_params": None if n_params is None else int(n_params),
        "model_size_kb": None if model_size_kb is None else float(model_size_kb),
        "notes": notes,
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{model_id}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(f"[{model_id}] accuracy={res['accuracy']:.4f}  F1 macro={res['f1_macro']:.4f}"
          + ("" if latency is None else f"  latencia={latency['single_ms']:.2f} ms/ventana"))
    return res


def _maybe_save(fig, save_name):
    if save_name:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")


def plot_confusion_matrix(cm, class_names, title: str = "", save_name: str | None = None, normalize=True):
    """Matriz de confusión (normalizada por fila si normalize=True)."""
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if shown[i, j] > 0.5 * shown.max() else "black", fontsize=8)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _maybe_save(fig, save_name)
    plt.show()
    return fig


def plot_history(history, title: str = "", save_name: str | None = None):
    """Curvas de pérdida y accuracy de un `History` de Keras."""
    import matplotlib.pyplot as plt
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"{title} — pérdida")
    axes[0].set_xlabel("época")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].set_xlabel("época")
    axes[1].legend()
    fig.tight_layout()
    _maybe_save(fig, save_name)
    plt.show()
    return fig
