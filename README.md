# Proyecto ML - Predicción del Estado Académico de Estudiantes

## Descripción

Este proyecto desarrolla un modelo de Machine Learning para predecir el estado académico de estudiantes universitarios.

El modelo permite clasificar a un estudiante en una de las siguientes categorías:

- Dropout
- Enrolled
- Graduate

El modelo entrenado se integra mediante una API REST desarrollada con FastAPI, permitiendo realizar predicciones a través del endpoint `/predict`.

---

## Objetivo

Desarrollar un modelo de Machine Learning capaz de predecir el estado académico de estudiantes y desplegarlo mediante una API REST utilizando FastAPI.

---

## Base de datos

Se utiliza el conjunto de datos:

**Predict Students' Dropout and Academic Success**

La base contiene:

- 4.424 estudiantes
- 37 variables
- 36 variables predictoras
- 1 variable objetivo: `Target`

La variable `Target` contiene tres categorías:

- `Dropout`
- `Enrolled`
- `Graduate`

### Distribución de la variable objetivo

| Categoría | Frecuencia | Porcentaje |
|---|---:|---:|
| Graduate | 2.209 | 49,93 % |
| Dropout | 1.421 | 32,12 % |
| Enrolled | 794 | 17,95 % |
| **Total** | **4.424** | **100 %** |

---

## Preparación de los datos

Antes del entrenamiento se realizó una revisión de la calidad de los datos.

### Valores faltantes

```text
0