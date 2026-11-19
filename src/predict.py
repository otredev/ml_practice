"""
Fase 7 — Predicción reproducible con explicación.

Carga el modelo y el umbral elegidos (models/config.json) y, para cada cliente, devuelve:
- probabilidad de abandono,
- decisión: "Contactar" / "No contactar" (según el umbral de costos),
- explicación: los factores que más suben y más bajan el riesgo.

Cómo se calcula la explicación (Regresión Logística):
  aporte de cada variable = coeficiente × (valor del cliente − valor promedio en train)
  Positivo → sube el riesgo respecto de un cliente promedio. Negativo → lo baja.
  Variables muy correlacionadas se agrupan para que la explicación tenga sentido:
  - TotalCharges se suma a Antigüedad (TotalCharges ≈ tenure × cargo mensual).
  - MonthlyCharges se suma a Internet (el cargo mensual depende sobre todo del tipo de internet).
  Sin agrupar, la colinealidad hacía aparecer cosas confusas como "un cargo mensual BAJO sube el riesgo".
  Si el modelo elegido fuera la red neuronal, la explicación se calcula igual con la
  Regresión Logística y es una aproximación.

Uso:
  python src/predict.py                                   # usa data/ejemplo_clientes.csv
  python src/predict.py --input mis_clientes.csv --output results/predicciones.csv

Desde Python (lo usa app.py):
  from predict import predecir
  predecir({"Contract": "Month-to-month", "tenure": 2, ...})
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import cargar_datos, limpiar, separar_datos

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RAIZ / "models"

# Nombres en español para la explicación
NOMBRES = {
    "tenure": "Antigüedad", "MonthlyCharges": "Cargo mensual", "TotalCharges": "Total facturado",
    "gender": "Género", "SeniorCitizen": "Adulto mayor", "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes", "PhoneService": "Servicio telefónico",
    "MultipleLines": "Múltiples líneas", "InternetService": "Internet",
    "OnlineSecurity": "Seguridad online", "OnlineBackup": "Backup online",
    "DeviceProtection": "Protección de equipos", "TechSupport": "Soporte técnico",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming películas",
    "Contract": "Contrato", "PaperlessBilling": "Factura digital", "PaymentMethod": "Forma de pago",
}
VALORES = {
    "Month-to-month": "mes a mes", "One year": "1 año", "Two year": "2 años",
    "Fiber optic": "fibra óptica", "DSL": "DSL", "Yes": "sí", "No": "no",
    "No internet service": "sin internet", "No phone service": "sin teléfono",
    "Electronic check": "cheque electrónico", "Mailed check": "cheque por correo",
    "Bank transfer (automatic)": "transferencia automática",
    "Credit card (automatic)": "tarjeta de crédito automática",
    "Male": "masculino", "Female": "femenino",
}
# Variables correlacionadas que se explican juntas (columna → grupo)
GRUPOS = {"TotalCharges": "tenure", "MonthlyCharges": "InternetService"}


@lru_cache(maxsize=1)
def _cargar():
    with open(RUTA_MODELOS / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    modelo = joblib.load(RUTA_MODELOS / f"{config['modelo']}.joblib")
    explicador = joblib.load(RUTA_MODELOS / "regresion_logistica.joblib")
    # Cliente "promedio" de train (ya transformado) como punto de comparación
    X, y = cargar_datos()
    X_train = separar_datos(X, y)[0]
    media_train = np.asarray(explicador.named_steps["prep"].transform(X_train).mean(axis=0)).ravel()
    return config, modelo, explicador, media_train


def _describir(columna, fila):
    valor = fila[columna]
    nombre = NOMBRES.get(columna, columna)
    if columna == "tenure":
        meses = int(valor)
        return f"{nombre}: {meses} {'mes' if meses == 1 else 'meses'}"
    if columna == "InternetService":
        return f"{nombre}: {VALORES.get(str(valor), str(valor))} (USD {float(fila['MonthlyCharges']):.2f}/mes)"
    return f"{nombre}: {VALORES.get(str(valor), str(valor))}"


def _explicar(fila: pd.DataFrame, explicador, media_train, n=3):
    prep = explicador.named_steps["prep"]
    coef = explicador.named_steps["modelo"].coef_[0]
    x = np.asarray(prep.transform(fila)).ravel()
    aportes = coef * (x - media_train)

    # Agrupar los aportes por variable original (una variable categórica tiene varias columnas one-hot)
    por_variable = {}
    for nombre_feature, aporte in zip(prep.get_feature_names_out(), aportes):
        tipo, resto = nombre_feature.split("__", 1)
        columna = resto if tipo == "num" else next(c for c in fila.columns if resto.startswith(c + "_") or resto == c)
        columna = GRUPOS.get(columna, columna)
        por_variable[columna] = por_variable.get(columna, 0.0) + aporte

    orden = sorted(por_variable.items(), key=lambda kv: kv[1], reverse=True)
    suben = [(_describir(c, fila.iloc[0]), round(a, 3)) for c, a in orden[:n] if a > 0]
    bajan = [(_describir(c, fila.iloc[0]), round(a, 3)) for c, a in orden[::-1][:n] if a < 0]
    return suben, bajan


def predecir(clientes):
    """clientes: dict (un cliente) o DataFrame (varios). Devuelve una lista de resultados."""
    config, modelo, explicador, media_train = _cargar()
    df = pd.DataFrame([clientes]) if isinstance(clientes, dict) else clientes.copy()
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]  # aproximación si no se informa
    if "Churn" not in df.columns:
        df["Churn"] = "No"  # columna requerida por limpiar(); no se usa para predecir
    ids = df["customerID"].tolist() if "customerID" in df.columns else list(range(1, len(df) + 1))
    X = limpiar(df).drop(columns=["Churn"])

    probas = modelo.predict_proba(X)[:, 1]
    resultados = []
    for i, p in enumerate(probas):
        suben, bajan = _explicar(X.iloc[[i]], explicador, media_train)
        decision = "Contactar" if p >= config["umbral"] else "No contactar"
        resultados.append({
            "cliente": ids[i],
            "probabilidad_abandono": round(float(p), 3),
            "decision": decision,
            "umbral": config["umbral"],
            "factores_que_suben_riesgo": [s for s, _ in suben],
            "factores_que_bajan_riesgo": [b for b, _ in bajan],
            "explicacion": _texto(p, decision, config["umbral"], suben, bajan),
        })
    return resultados


def _texto(p, decision, umbral, suben, bajan):
    texto = f"Probabilidad de abandono: {p:.0%}. "
    if decision == "Contactar":
        texto += f"Supera el umbral de {umbral:.0%}, así que conviene contactarlo. "
    else:
        texto += f"Está por debajo del umbral de {umbral:.0%}, así que no hace falta contactarlo. "
    if suben:
        texto += "Lo que más aumenta el riesgo: " + "; ".join(s for s, _ in suben) + ". "
    if bajan:
        texto += "Lo que más lo reduce: " + "; ".join(b for b, _ in bajan) + "."
    return texto.strip()


def main():
    parser = argparse.ArgumentParser(description="Predice el abandono de clientes.")
    parser.add_argument("--input", default=str(RAIZ / "data" / "ejemplo_clientes.csv"))
    parser.add_argument("--output", default=str(RAIZ / "results" / "predicciones.csv"))
    args = parser.parse_args()

    resultados = predecir(pd.read_csv(args.input))
    for r in resultados:
        print(f"[{r['cliente']}] {r['decision'].upper()} — {r['explicacion']}\n")

    salida = pd.DataFrame(resultados)
    for col in ["factores_que_suben_riesgo", "factores_que_bajan_riesgo"]:
        salida[col] = salida[col].apply(" | ".join)
    salida.to_csv(args.output, index=False)
    print(f"Guardado: {Path(args.output).relative_to(RAIZ) if Path(args.output).is_relative_to(RAIZ) else args.output}")


if __name__ == "__main__":
    main()
