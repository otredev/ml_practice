"""
Fase 3 — Entrenamiento de modelos.

Modelos:
1. Regla simple (sin ML, referencia): contactar a todos los clientes con contrato mes a mes.
2. Regresión Logística: modelo lineal, simple y explicable.
3. Red neuronal (MLPClassifier): modelo no lineal, obligatorio en la consigna.

Los dos modelos de ML se entrenan dentro de un Pipeline (preprocesador + modelo),
ajustado SOLO con train. Se guardan en models/ para usarlos en evaluate.py, predict.py y app.py.

Ejecutar: python src/train.py
"""
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import RANDOM_STATE, cargar_datos, crear_preprocesador, separar_datos

RUTA_MODELOS = Path(__file__).resolve().parent.parent / "models"


def regla_simple(X):
    """Referencia sin ML: 1 (contactar) si el contrato es mes a mes, 0 si no."""
    return (X["Contract"] == "Month-to-month").astype(int).to_numpy()


def crear_modelos():
    return {
        "regresion_logistica": Pipeline([
            ("prep", crear_preprocesador()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "red_neuronal": Pipeline([
            ("prep", crear_preprocesador()),
            ("modelo", MLPClassifier(
                hidden_layer_sizes=(32, 16),  # 2 capas ocultas: 32 y 16 neuronas
                activation="relu",
                alpha=0.001,                  # regularización L2: castiga pesos grandes (anti-overfitting)
                early_stopping=True,          # aparta 10 % de train y frena si deja de mejorar
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=500,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    X, y = cargar_datos(verbose=True)
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y, verbose=True)
    RUTA_MODELOS.mkdir(exist_ok=True)

    print("\nROC-AUC en train vs validación (si train >> validación hay overfitting):")
    for nombre, pipeline in crear_modelos().items():
        pipeline.fit(X_train, y_train)  # el preprocesador se ajusta aquí, solo con train
        auc_train = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
        auc_val = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        print(f"  {nombre:<20} train {auc_train:.3f} | validación {auc_val:.3f} | diferencia {auc_train - auc_val:+.3f}")
        if nombre == "red_neuronal":
            print(f"  {'':<20} (la red se detuvo en la época {pipeline.named_steps['modelo'].n_iter_})")
        joblib.dump(pipeline, RUTA_MODELOS / f"{nombre}.joblib")

    print("\nModelos guardados en models/")


if __name__ == "__main__":
    main()
