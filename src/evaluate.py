"""
Fase 4 — Evaluación de modelos en VALIDACIÓN (umbral por defecto 0,5).

Compara:
- Estrategias extremas: no contactar a nadie / contactar a todos.
- Regla simple sin ML: contactar a contratos mes a mes.
- Regresión Logística y Red neuronal (modelos guardados por train.py).

Reporta: matriz de confusión, precision, recall, F1, ROC-AUC, accuracy y costo en USD.
El test NO se usa aquí: se reserva para la evaluación final.

Ejecutar: python src/evaluate.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import cargar_datos, separar_datos
from train import regla_simple

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RAIZ / "models"
RUTA_RESULTADOS = RAIZ / "results"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"

COSTO_FN = 120  # no detectar a un cliente que se va
COSTO_FP = 15   # contactar a un cliente que no se iba
MODELOS = {"Regresión Logística": "regresion_logistica", "Red neuronal": "red_neuronal"}


def costo_total(y_true, y_pred):
    """Costo de negocio en USD = FN × 120 + FP × 15."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * COSTO_FN + fp * COSTO_FP


def metricas(nombre, y_true, y_pred, y_proba=None, umbral=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "modelo": nombre,
        "umbral": umbral,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "costo_usd": fn * COSTO_FN + fp * COSTO_FP,
    }


def cargar_modelos():
    return {nombre: joblib.load(RUTA_MODELOS / f"{archivo}.joblib") for nombre, archivo in MODELOS.items()}


def main():
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y)
    modelos = cargar_modelos()
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    filas = [
        metricas("No contactar a nadie", y_val, np.zeros(len(y_val), dtype=int)),
        metricas("Contactar a todos", y_val, np.ones(len(y_val), dtype=int)),
        metricas("Regla: contrato mes a mes", y_val, regla_simple(X_val)),
    ]
    probas = {}
    for nombre, modelo in modelos.items():
        probas[nombre] = modelo.predict_proba(X_val)[:, 1]
        y_pred = (probas[nombre] >= 0.5).astype(int)
        filas.append(metricas(nombre, y_val, y_pred, probas[nombre], umbral=0.5))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_RESULTADOS / "metricas_validacion.csv", index=False)
    pd.set_option("display.width", 200)
    print("Métricas en VALIDACIÓN (umbral 0,5 para los modelos):\n")
    print(tabla.drop(columns="umbral").round(3).to_string(index=False))

    # Matrices de confusión
    casos = {"Regla: contrato mes a mes": regla_simple(X_val)}
    casos.update({f"{n} (umbral 0,5)": (p >= 0.5).astype(int) for n, p in probas.items()})
    fig, axes = plt.subplots(1, len(casos), figsize=(5 * len(casos), 4))
    for ax, (titulo, y_pred) in zip(axes, casos.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred),
                               display_labels=["Se queda", "Abandona"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{titulo}\nCosto: USD {costo_total(y_val, y_pred):,}".replace(",", "."))
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(RUTA_FIGURAS / "05_matrices_confusion_validacion.png", dpi=120)
    plt.close()

    # Curva ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, p in probas.items():
        RocCurveDisplay.from_predictions(y_val, p, name=nombre, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0,5)")
    ax.set_title("Curva ROC — validación")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RUTA_FIGURAS / "06_curva_roc_validacion.png", dpi=120)
    plt.close()

    print("\nGuardado: results/metricas_validacion.csv y figuras 05 y 06 en results/figuras/")


if __name__ == "__main__":
    main()
