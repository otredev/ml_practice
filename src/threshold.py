"""
Fase 5 — Elección del umbral de decisión según los costos (en VALIDACIÓN).

Un modelo devuelve una probabilidad de abandono. El umbral decide a partir de qué
probabilidad contactamos al cliente. 0,5 no tiene por qué ser el mejor: un falso negativo
(USD 120) cuesta 8 veces más que un falso positivo (USD 15).

Referencia teórica: conviene contactar si  p × 120 > (1 − p) × 15  →  p > 15 / 135 ≈ 0,11
(suponiendo que el contacto siempre retiene al cliente y que las probabilidades están bien calibradas).

Se prueban umbrales de 0,01 a 0,99 y se elige el de menor costo EN VALIDACIÓN.
El modelo final es el de menor costo; si la diferencia es menor al 2 %, se prefiere la
Regresión Logística por ser más simple y explicable.
Resultado: models/config.json (modelo y umbral elegidos), usado por predict.py y app.py.

Ejecutar: python src/threshold.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import (COSTO_FN, COSTO_FP, MODELOS, RUTA_FIGURAS, RUTA_MODELOS, RUTA_RESULTADOS,
                      cargar_modelos, costo_total, metricas)
from preprocessing import cargar_datos, separar_datos
from train import regla_simple

UMBRALES = np.round(np.arange(0.01, 1.00, 0.01), 2)
TOLERANCIA = 0.02  # 2 %: diferencias menores se consideran empate


def main():
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y)
    modelos = cargar_modelos()

    costo_regla = costo_total(y_val, regla_simple(X_val))
    costo_todos = costo_total(y_val, np.ones(len(y_val), dtype=int))

    curvas, mejores = {}, []
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_val)[:, 1]
        costos = np.array([costo_total(y_val, (proba >= u).astype(int)) for u in UMBRALES])
        curvas[nombre] = costos
        u_opt = float(UMBRALES[costos.argmin()])
        fila = metricas(nombre, y_val, (proba >= u_opt).astype(int), proba, umbral=u_opt)
        fila["costo_umbral_0.5"] = costo_total(y_val, (proba >= 0.5).astype(int))
        mejores.append(fila)

    tabla = pd.DataFrame(mejores)
    tabla.to_csv(RUTA_RESULTADOS / "umbral_optimo_validacion.csv", index=False)
    pd.set_option("display.width", 200)
    print("Umbral óptimo por modelo (VALIDACIÓN):\n")
    print(tabla[["modelo", "umbral", "precision", "recall", "f1", "FN", "FP",
                 "costo_usd", "costo_umbral_0.5"]].round(3).to_string(index=False))
    print(f"\nReferencias: regla mes a mes USD {costo_regla:,} | contactar a todos USD {costo_todos:,}")

    # Elección del modelo final (solo con validación)
    tabla = tabla.sort_values("costo_usd").reset_index(drop=True)
    elegido = tabla.iloc[0]
    rl = tabla[tabla["modelo"] == "Regresión Logística"].iloc[0]
    if elegido["modelo"] != "Regresión Logística" and (rl["costo_usd"] - elegido["costo_usd"]) / elegido["costo_usd"] < TOLERANCIA:
        print(f"\n{elegido['modelo']} cuesta apenas menos (< {TOLERANCIA:.0%}): se prefiere la Regresión Logística por simplicidad.")
        elegido = rl
    config = {
        "modelo": MODELOS[elegido["modelo"]],
        "nombre_modelo": elegido["modelo"],
        "umbral": float(elegido["umbral"]),
        "costo_validacion_usd": int(elegido["costo_usd"]),
        "costo_fn": COSTO_FN,
        "costo_fp": COSTO_FP,
        # Umbral óptimo de cada modelo (para poder cambiar de modelo en la interfaz)
        "modelos": {MODELOS[r["modelo"]]: {"nombre": r["modelo"], "umbral": float(r["umbral"]),
                                           "costo_validacion_usd": int(r["costo_usd"])}
                    for _, r in tabla.iterrows()},
    }
    with open(RUTA_MODELOS / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\nElegido: {config['nombre_modelo']} con umbral {config['umbral']} "
          f"(costo en validación USD {config['costo_validacion_usd']:,}) → models/config.json")

    # Gráfico: costo vs umbral
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre, costos in curvas.items():
        u_opt = UMBRALES[costos.argmin()]
        ax.plot(UMBRALES, costos, label=f"{nombre} (mínimo USD {costos.min():,.0f} en {u_opt:.2f})")
        ax.scatter([u_opt], [costos.min()], zorder=3)
    ax.axhline(costo_regla, color="gray", linestyle="--", label=f"Regla mes a mes (USD {costo_regla:,})")
    ax.axhline(costo_todos, color="gray", linestyle=":", label=f"Contactar a todos (USD {costo_todos:,})")
    ax.axvline(0.5, color="black", linewidth=0.8, alpha=0.4)
    ax.text(0.505, ax.get_ylim()[1] * 0.95, "umbral 0,5", fontsize=8)
    ax.set_xlabel("Umbral de decisión (probabilidad mínima para contactar)")
    ax.set_ylabel("Costo total en validación (USD)")
    ax.set_title("Costo de negocio según el umbral — validación")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RUTA_FIGURAS / "07_costo_vs_umbral.png", dpi=120)
    plt.close()
    print("Gráfico: results/figuras/07_costo_vs_umbral.png")


if __name__ == "__main__":
    main()
