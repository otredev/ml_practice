# Contexto del proyecto: predicción de abandono de clientes (Telco)

Este documento describe el problema, los datos y lo que se espera del proyecto.
No contiene decisiones técnicas: esas se toman y se justifican durante el desarrollo.

## 1. La empresa y el problema

Una empresa de telecomunicaciones está perdiendo clientes: algunos abandonan el servicio
y se van a otra compañía. La empresa quiere usar inteligencia artificial para **anticiparse**
a esos abandonos y tomar mejores decisiones comerciales.

Somos el equipo de tecnología contratado para desarrollar la solución.

**Misión:** construir un sistema que, a partir de los datos de un cliente, estime la
**probabilidad de que abandone** la empresa.

Una predicción por sí sola no mejora el negocio: el valor está en la **decisión** que
se toma con ella (a quién contactar para intentar retenerlo).

## 2. Los datos

- Dataset público **Telco Customer Churn** (Kaggle):
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn. 
  - El archivo está en `archive/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Contiene información de cada cliente: servicios contratados, gastos, antigüedad
  y si abandonó o no la compañía (`Churn`).
- Antes de construir nada hay que entender qué información está disponible, qué
  problemas de calidad tiene y qué problemas podrían surgir al usarla.

## 3. La decisión de negocio

La empresa estima estos costos (supuestos del ejercicio):

| Error | Qué significa | Costo estimado |
|---|---|---|
| Falso negativo | No detectar a un cliente que finalmente abandona | USD 120 |
| Falso positivo | Contactar innecesariamente a un cliente que no pensaba irse | USD 15 |

Además de comparar la calidad predictiva de los modelos, hay que usar estos costos
para **recomendar una estrategia**.

**Pregunta central:** ¿el modelo con mejores métricas es necesariamente el más
conveniente para la empresa?

## 4. Qué debe lograr la solución

1. **Comprender los datos:** identificar información relevante, problemas de calidad
   y posibles patrones asociados al abandono.
2. **Realizar predicciones:** recibir datos de clientes y estimar quiénes podrían
   abandonar la compañía.
3. **Comparar alternativas:** experimentar con al menos dos enfoques de Machine
   Learning, uno de ellos obligatoriamente una **red neuronal**.
4. **Evaluar resultados:** presentar evidencia cuantitativa del desempeño de cada
   modelo y reconocer sus limitaciones.
5. **Recomendar una decisión:** proponer qué modelo debería usar la empresa y
   justificar la elección.

No existe una única solución correcta: lo importante es que las decisiones tengan sentido.

## 5. Tu rol

Eres un **asistente de programación, no el responsable de las decisiones**.
Puedes ayudar a:

- Explorar y comprender el dataset.
- Diseñar e implementar soluciones.
- Explicar conceptos o fragmentos de código.
- Detectar y corregir errores.
- Crear pruebas, analizar resultados y proponer mejoras.

Nosotros seremos los responsables de explicar qué construyes, cómo funciona y por qué elegimos ese
enfoque. Debes explicar tus propuestas y supuestos para que el equipo pueda validarlos, cuestionarlos o corregirlos.

## 6. Repositorio de GitHub

El link del repo es https://github.com/otredev/ml_practice

Reglas para actualizar el README
- Instrucciones claras para instalar dependencias y ejecutar el proyecto.
- Listar los notebooks y scripts de análisis y entrenamiento, con qué hace cada uno.
- Explicar cómo generar predicciones de forma reproducible.
- Incluir resultados, gráficos y comparaciones relevantes.
- Otra persona debe poder clonar el repo, seguir el README y reproducir el experimento.

## 7. Cómo se evalúa el proyecto

- No gana el porcentaje de aciertos más alto: se valora más un modelo sencillo,
  bien evaluado y comprendido que uno complejo que no se pueda explicar.
- Se evalúa la justificación de decisiones técnicas y comerciales (costos USD 120 / USD 15).
- Todo debe ser reproducible: otra persona debe poder clonar el repo y obtener los mismos resultados.

