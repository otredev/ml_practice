# Predicción de abandono de clientes (Churn) — Telco

Sistema que estima la probabilidad de que un cliente de una empresa de telecomunicaciones
abandone el servicio, comparando una Regresión Logística y una Red Neuronal, y que
recomienda a qué clientes contactar según los costos del negocio
(falso negativo = USD 120, falso positivo = USD 15).

## Reglas para actualizar este README
- Instrucciones claras para instalar dependencias y ejecutar el proyecto.
- Listar los notebooks y scripts de análisis y entrenamiento, con qué hace cada uno.
- Explicar cómo generar predicciones de forma reproducible.
- Incluir resultados, gráficos y comparaciones relevantes.
- Otra persona debe poder clonar el repo, seguir el README y reproducir el experimento.

## Instalación
```bash
git clone <URL_DEL_REPO>
cd ml_practice
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
ubicado en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Ejecución
```bash
python src/preprocessing.py   # Fase 2: limpieza y partición train/validación/test (muestra un resumen)
python src/train.py           # Fase 3: entrena Regresión Logística y Red Neuronal, guarda en models/
python src/evaluate.py        # Fase 4: métricas en validación, matrices de confusión y curva ROC
python src/threshold.py       # Fase 5: umbral óptimo según costos → models/config.json
python src/final_test.py      # Fase 6: evaluación final en test (una sola vez)
```
_Completar con los siguientes comandos (entrenamiento, evaluación, predicción)._

## Estructura del proyecto
```
data/              dataset original
notebooks/         análisis exploratorio (EDA)
src/               preprocesamiento, entrenamiento, evaluación y predicción
models/            modelos entrenados
results/           tablas de resultados
results/figuras/   gráficos
```

## Generar predicciones
_Completar: cómo usar `src/predict.py`, con un ejemplo de entrada y salida._

## Resultados
_Completar: tabla comparativa (precision, recall, F1, ROC-AUC, costo USD) y gráficos._

## Recomendación
**Regresión Logística con umbral 0,14**: contactar a todo cliente con probabilidad de abandono mayor al 14 %.

| Estrategia (test, 1.409 clientes) | Recall | FN | FP | Costo (USD) |
|---|---|---|---|---|
| Regresión Logística, umbral 0,14 (elegida) | 0,917 | 31 | 457 | **10.575** |
| Regresión Logística, umbral 0,5 | 0,527 | 177 | 122 | 23.070 |
| Regla: contrato mes a mes | 0,866 | 50 | 462 | 12.930 |
| Contactar a todos | 1,000 | 0 | 1.035 | 15.525 |

Ahorra 18 % frente a la regla simple. La red neuronal costó prácticamente lo mismo en validación,
así que se eligió la Regresión Logística por ser más simple y explicable.

## Equipo
- 
