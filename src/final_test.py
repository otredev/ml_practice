"""
Fase 6 — Evaluación final en TEST (se usa una sola vez).

Toma el modelo y el umbral ya elegidos en validación (models/config.json) y mide su
desempeño real en datos que nunca se usaron para decidir nada.
Las demás estrategias se muestran solo como referencia: la decisión ya está tomada
y NO se cambia según estos resultados.

Ejecutar: python src/final_test.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import RUTA_FIGURAS, RUTA_MODELOS, RUTA_RESULTADOS, cargar_modelos, metricas
from preprocessing import cargar_datos, separar_datos
from train import regla_simple


def main():
    with open(RUTA_MODELOS / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    nombre, umbral = config["nombre_modelo"], config["umbral"]

    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y)
    modelos = cargar_modelos()

    proba = modelos[nombre].predict_proba(X_test)[:, 1]
    y_pred = (proba >= umbral).astype(int)
    final = metricas(f"{nombre} (umbral {umbral}) — ELEGIDO", y_test, y_pred, proba, umbral)

    # Referencias (no cambian la decisión)
    filas = [
        final,
        metricas(f"{nombre} (umbral 0,5)", y_test, (proba >= 0.5).astype(int), proba, 0.5),
        metricas("Regla: contrato mes a mes", y_test, regla_simple(X_test)),
        metricas("Contactar a todos", y_test, np.ones(len(y_test), dtype=int)),
        metricas("No contactar a nadie", y_test, np.zeros(len(y_test), dtype=int)),
    ]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RUTA_RESULTADOS / "resultado_test.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Clientes en test: {len(y_test)} ({y_test.sum()} abandonan)\n")
    print(tabla.drop(columns="umbral").round(3).to_string(index=False))
    print(f"\nCosto del modelo elegido: validación USD {config['costo_validacion_usd']:,} "
          f"→ test USD {final['costo_usd']:,}")
    print(f"Costo por cliente en test: USD {final['costo_usd'] / len(y_test):.2f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["Se queda", "Abandona"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"TEST — {nombre} (umbral {umbral})\nCosto: USD {final['costo_usd']:,}".replace(",", "."))
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(RUTA_FIGURAS / "08_matriz_confusion_test.png", dpi=120)
    plt.close()
    print("\nGuardado: results/resultado_test.csv y results/figuras/08_matriz_confusion_test.png")


if __name__ == "__main__":
    main()
