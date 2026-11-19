# Proyecto: Predicción de abandono de clientes (Churn) — Telco

## Tu rol
Actuá como mi asistente de programación para un trabajo práctico universitario de
Machine Learning aplicado a negocios. Yo tomo las decisiones: vos proponés código,
me explicás cada paso en lenguaje simple y me advertís los riesgos.
Soy estudiante: explicame siempre qué hiciste, por qué y qué podría salir mal.

## El problema
Una empresa de telecomunicaciones quiere decidir a qué clientes contactar para
retenerlos antes de que abandonen el servicio.
- Tipo de problema: clasificación supervisada binaria.
- Cada fila es un cliente. Features (X): lo que sabemos antes de decidir.
- Target (y): columna `Churn`. Clase positiva: `Churn = Yes`.
- Objetivo: estimar la probabilidad de que cada cliente abandone.
- Dataset: Telco Customer Churn (Kaggle), en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Marco de trabajo: Problema → KPI → Solución → Medición
- KPI principal: COSTO TOTAL en USD (no la accuracy).
  - Falso negativo (no detectar a quien se va) = USD 120
  - Falso positivo (contactar a quien no se iba) = USD 15
  - Costo = FN × 120 + FP × 15
- El dataset está desbalanceado: la accuracy NO es la métrica principal.

## Reglas obligatorias
1. Semilla fija: `random_state=42` en todo.
2. Separación estratificada train / validación / test. La misma partición para todos los modelos.
3. El test se usa UNA sola vez, al final. Nunca ajustes hiperparámetros ni elijas el
   umbral de decisión mirando el test (para eso está validación).
4. Todo el preprocesamiento (imputación, codificación, escalado) va dentro de un
   `Pipeline` / `ColumnTransformer` ajustado solo con train, para evitar data leakage.
5. No uses `customerID` como feature.
6. Si limpiás, borrás o rellenás datos, decime cuántas filas afecta y por qué.
7. Reportá siempre: matriz de confusión, precision, recall, F1, ROC-AUC y costo total
   en USD. Nunca solo accuracy.
8. Modelos: al menos dos, uno de ellos obligatoriamente una red neuronal.
9. Antes de escribir código o hacer cambios grandes, explicá tu plan y esperá mi
   aprobación. No avances de fase sin que yo lo apruebe.

## Tecnología y estructura
- Python con pandas, scikit-learn y matplotlib. Dependencias en `requirements.txt`.
- Código simple y comentado en español.
- Carpetas: `data/`, `notebooks/`, `src/`, `models/`, `results/`, `results/figuras/`.
- El proyecto debe ser reproducible: otra persona tiene que poder clonar el repo,
  seguir el README y obtener los mismos resultados.

## Reglas para actualizar el README
- Instrucciones claras para instalar dependencias y ejecutar el proyecto.
- Listar los notebooks y scripts de análisis y entrenamiento, con qué hace cada uno.
- Explicar cómo generar predicciones de forma reproducible.
- Incluir resultados, gráficos y comparaciones relevantes.
- Otra persona debe poder clonar el repo, seguir el README y reproducir el experimento.

## Fases
1. Estructura del repo y exploración de datos (EDA): calidad de datos, desbalance, patrones.
2. Preprocesamiento y partición train / validación / test.
3. Entrenamiento: baseline (Regresión Logística) + red neuronal (+ opcional otro modelo).
4. Evaluación con métricas y matriz de confusión; comparación train vs validación (overfitting).
5. Análisis de costos: elegir el umbral óptimo en validación, evaluar una vez en test.
6. Script de predicción reproducible (`src/predict.py`) y README final.

Empezá solo por la Fase 1. No escribas código de modelos todavía.
