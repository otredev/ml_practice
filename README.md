# Predicción de abandono de clientes Telco

Proyecto para estimar el abandono de clientes y comparar alternativas considerando
los costos del ejercicio: USD 120 por falso negativo y USD 15 por falso positivo.
El contexto completo está en [CONTEXT.md](CONTEXT.md).

## Preparación

La limpieza utiliza únicamente la biblioteca estándar de Python (3.9 o posterior).
Para los notebooks, el entorno verificado usa **Python 3.14** y las versiones
fijadas en `requirements.txt`.

```bash
git clone https://github.com/otredev/ml_practice.git
cd ml_practice
python --version
```

En Windows también se puede usar `py` en lugar de `python`.

Para instalar las dependencias de los notebooks en un entorno aislado:

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Con el entorno activo:

```bash
python -m pip install -r requirements.txt
```

## Reproducir la limpieza

Desde la raíz del repositorio:

```bash
python src/limpieza.py
```

El script lee `archive/WA_Fn-UseC_-Telco-Customer-Churn_raw.csv` y genera
`archive/WA_Fn-UseC_-Telco-Customer-Churn_clean.csv`. Si el archivo limpio ya
existe, lo reemplaza. El raw permanece intacto. Las rutas predeterminadas se
resuelven respecto al repositorio, independientemente del directorio de ejecución.

Reglas aplicadas:

- Conserva `customerID` para identificar clientes. Se excluirá de las features
  únicamente al entrenar.
- Interpreta `TotalCharges` como importe decimal, con punto como separador.
  Los espacios se consideran faltantes. Un faltante se reemplaza por `0.00`
  **solo cuando `tenure = 0`**. Los importes existentes se conservan, incluso
  cuando la antigüedad es cero.
- Si falta `TotalCharges` y `tenure > 0`, conserva el campo vacío e informa cuántos
  casos requieren revisión. No aplica otra imputación automática.
- Convierte `SeniorCitizen`: `0` a `No` y `1` a `Yes`.
- Conserva `No internet service` y `No phone service` como categorías propias.
- Conserva el orden de filas y columnas y los demás valores. Rechaza importes o
  antigüedades inválidos y valores de `SeniorCitizen` distintos de 0/1 antes de
  escribir la salida.

Un CSV no almacena tipos de datos: los importes se escriben como números decimales
y la herramienta que los lea deberá interpretarlos como numéricos. Los faltantes
pendientes se representan mediante campos vacíos.

Con el raw actual, la salida tiene **7.043 filas y 21 columnas**, con **11 importes
imputados a cero** y **ningún `TotalCharges` pendiente**.

Para usar otras rutas:

```bash
python src/limpieza.py --entrada ruta/datos_raw.csv --salida ruta/datos_clean.csv
```

La carpeta de salida debe existir.

## Separación y estrategias sin modelo

Después de la limpieza y con el entorno activo, ejecutar en este orden:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_separacion.ipynb --ExecutePreprocessor.timeout=180
python -m nbconvert --to notebook --execute --inplace notebooks/02_baselines_validacion.ipynb --ExecutePreprocessor.timeout=180
```

También se pueden abrir con `python -m jupyterlab` y ejecutar todas las celdas,
primero del notebook 01 y después del 02, usando el kernel del entorno `.venv`.
Los notebooks se guardan con las salidas de su ejecución.

### Separación persistida

El notebook 01 aplica `train_test_split` en dos pasos: reserva el 30 % y divide
esa reserva en mitades. Ambos pasos usan `stratify=Churn` y `random_state=42`.
El redondeo a filas enteras produce:

| Conjunto | Filas | Uso |
|---|---:|---|
| Train | 4.930 | Ajustar futuros modelos y transformaciones |
| Validación | 1.056 | Comparar estrategias y elegir futuros modelos/umbrales |
| Test | 1.057 | Reservado para la evaluación final; no se evalúa en este paso |

Los archivos se guardan en `data/splits/`:

- `train.csv`, `validacion.csv` y `test.csv`: datos completos de cada conjunto,
  conservando `customerID` y `Churn`.
- `asignacion.csv`: relación `customerID,conjunto`, para reutilizar la separación
  incluso si cambia el orden de las filas.
- `metadata.json`: semilla, método, tamaños, versiones y huellas SHA-256 de la
  fuente y de los archivos generados.

Al repetir el notebook 01, se reutiliza la separación existente. Si cambia el CSV
limpio, se detiene para que se revise explícitamente la versión de los datos.
Todos los modelos deberán cargar estos mismos archivos, sin volver a separar.
`customerID` se conserva para trazabilidad y debe excluirse de las features, junto
con `Churn` que se utiliza como objetivo. Cualquier transformación que aprenda de
los datos se ajustará exclusivamente con train.

### Evaluación en validación

El notebook 02 carga **únicamente validación**, verifica su huella contra la
separación guardada y evalúa dos decisiones constantes. No entrena modelos ni
carga `test.csv`.

La clase positiva es `Churn=Yes` y la predicción positiva significa contactar.
Las matrices tienen filas reales `[No, Yes]` y columnas de decisión
`[No contactar, Contactar]`: `[[TN, FP], [FN, TP]]`.
El costo se calcula como `120 * FN + 15 * FP` USD, con costo cero para TP y TN
según los supuestos del ejercicio. No se supone que contactar garantice retener.

Los resultados se guardan en `outputs/baselines_validacion/metricas.csv`, junto
con un `metadata.json` que documenta costos, convención de precisión y huella de
validación. Las matrices y las métricas también quedan visibles en el notebook.

Para “no contactar a nadie”, precision no está definida al no haber predicciones
positivas; se registra 0 por convención (`zero_division=0`). Accuracy, precision y
recall se guardan como proporciones entre 0 y 1.

Resultados sobre los 1.056 clientes de validación (280 con abandono):

| Estrategia | TN | FP | FN | TP | Accuracy | Precision | Recall | Costo total USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No contactar a nadie | 776 | 0 | 280 | 0 | 73,48 % | 0 %* | 0 % | 33.600 |
| Contactar a todos | 0 | 776 | 0 | 280 | 26,52 % | 26,52 % | 100 % | 11.640 |

\* Valor convencional: no hay predicciones positivas.

Matrices: no contactar a nadie `[[776, 0], [280, 0]]`; contactar a todos
`[[0, 776], [0, 280]]`. Contactar a todos tiene menor costo por **USD 21.960**
con estos supuestos, pese a su menor accuracy. Estas estrategias son referencias
para los futuros modelos, no una estimación del efecto real de contactar clientes.

## Scripts y pruebas

Para regenerar los dos informes de una página, después de ejecutar ambos notebooks:

```bash
python src/exploracion_pdf.py
```

Genera `reports/01_exploracion.pdf` con abandono general, por contrato, por
antigüedad y por tipo de internet. Los gráficos usan exclusivamente train para
mantener reservado test; la auditoría de calidad del raw no analiza sus etiquetas
de abandono. El script verifica la huella de train, documenta las decisiones de
limpieza y no entrena modelos. Requiere `reportlab`, incluido en `requirements.txt`.

También genera `reports/02_baseline_validacion.pdf`: tabla de métricas, gráfico
de costos, matrices de confusión y una línea de conclusión sobre las dos
estrategias. Verifica los resultados guardados contra las etiquetas y la huella
de validación. No abre ni modifica el conjunto de test.

Para regenerar únicamente uno de los informes:

```bash
python src/exploracion_pdf.py --informe exploracion
python src/exploracion_pdf.py --informe baseline
```

| Archivo | Función |
|---|---|
| `src/limpieza.py` | Genera el CSV limpio a partir del raw e informa los cambios. |
| `src/exploracion_pdf.py` | Regenera los PDFs de exploración y baselines de validación. |
| `notebooks/01_separacion.ipynb` | Crea o reutiliza la separación estratificada 70/15/15. |
| `notebooks/02_baselines_validacion.ipynb` | Evalúa las dos estrategias sin modelo, solo en validación. |
| `tests/test_limpieza.py` | Comprueba las reglas de imputación, conservación de datos y protección del raw. |
| `tests/test_experimento.py` | Comprueba asignación, estratificación de train/validación y métricas por conteo independiente; nunca abre `test.csv`. |

```bash
python -m unittest discover -s tests -v
```

Las pruebas del experimento requieren ejecutar primero ambos notebooks; si aún
no existen sus resultados, se omiten esas pruebas. Todavía no hay entrenamiento
ni predicciones de modelos aprendidos. Las instrucciones para reproducir esos
experimentos y sus resultados se incorporarán cuando se implementen.
