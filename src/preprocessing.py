"""
Fase 2 — Preparación de datos.

Funciones reutilizables por train.py, evaluate.py, predict.py y app.py:
- cargar_datos():      lee el CSV y hace la limpieza básica.
- separar_datos():     divide en train / validación / test (60/20/20) estratificado.
- crear_preprocesador(): ColumnTransformer (one-hot + escalado) SIN ajustar.

Regla clave: el preprocesador se ajusta (fit) SOLO con train, dentro de un Pipeline.
Así evitamos data leakage (que información de validación/test se filtre al modelo).
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
COLUMNAS_A_ELIMINAR = ["customerID"]  # identificador: no aporta información
COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
COLUMNAS_CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def limpiar(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Limpieza que no 'aprende' nada de los datos (se puede aplicar antes de separar)."""
    df = df.copy()

    # TotalCharges viene como texto y tiene espacios vacíos en clientes con tenure = 0.
    # Son clientes nuevos a los que todavía no se les cobró: rellenamos con 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_vacios = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if verbose:
        print(f"TotalCharges: {n_vacios} valores vacíos rellenados con 0 (clientes con tenure = 0).")

    # SeniorCitizen es 0/1 pero en realidad es una categoría (sí/no).
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df.drop(columns=[c for c in COLUMNAS_A_ELIMINAR if c in df.columns])


def cargar_datos(ruta: Path = RUTA_DATOS, verbose: bool = False):
    """Devuelve X (features) e y (1 = abandona, 0 = se queda)."""
    df = limpiar(pd.read_csv(ruta), verbose=verbose)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    if verbose:
        print(f"Datos: {X.shape[0]} clientes, {X.shape[1]} features. Tasa de churn: {y.mean():.1%}")
    return X, y


def separar_datos(X, y, verbose: bool = False):
    """Train 60 % / validación 20 % / test 20 %, estratificado y con semilla fija.

    - train: para entrenar los modelos.
    - validación: para comparar modelos y elegir el umbral de decisión.
    - test: se usa UNA sola vez al final.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    if verbose:
        for nombre, yy in [("train", y_train), ("validación", y_val), ("test", y_test)]:
            print(f"{nombre:>11}: {len(yy):>5} clientes | churn {yy.mean():.1%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def crear_preprocesador() -> ColumnTransformer:
    """One-hot para categóricas y escalado para numéricas. Se devuelve SIN ajustar:
    el ajuste ocurre dentro del Pipeline al hacer .fit() con train."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUMNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), COLUMNAS_CATEGORICAS),
        ]
    )


if __name__ == "__main__":
    # Ejecutar: python src/preprocessing.py  → muestra un resumen de la preparación
    X, y = cargar_datos(verbose=True)
    X_train, X_val, X_test, y_train, y_val, y_test = separar_datos(X, y, verbose=True)
    prep = crear_preprocesador().fit(X_train)  # ajustado SOLO con train
    print(f"Features después del preprocesamiento: {len(prep.get_feature_names_out())}")
