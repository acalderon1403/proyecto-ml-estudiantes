# Proyecto ML - Predicción del Estado Académico de Estudiantes

## Descripción

Este proyecto desarrolla un modelo de Machine Learning para predecir el estado académico de estudiantes universitarios.

El modelo clasifica a cada estudiante en una de las siguientes categorías:

- **Dropout:** estudiante que abandona sus estudios.
- **Enrolled:** estudiante que continúa matriculado.
- **Graduate:** estudiante que se gradúa.

El modelo se integra mediante una API REST desarrollada con FastAPI.

## Dataset

Se utiliza el dataset **Predict Students' Dropout and Academic Success**.

La base contiene:

- **4424 observaciones**
- **37 variables**
- **36 variables predictoras**
- **1 variable objetivo:** `Target`
- **0 valores faltantes**
- **0 registros duplicados**

### Distribución de la variable objetivo

| Estado | Frecuencia | Porcentaje |
|---|---:|---:|
| Graduate | 2209 | 49.93% |
| Dropout | 1421 | 32.12% |
| Enrolled | 794 | 17.95% |

## Metodología

Se utilizó una división de los datos mediante Holdout:

- **80%** para entrenamiento.
- **20%** para prueba.
- **3539 observaciones** para entrenamiento.
- **885 observaciones** para prueba.

Se evaluaron dos modelos:

1. Regresión Logística.
2. Random Forest.

### Resultados

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.5672 | 0.6169 | 0.5672 | 0.5837 |
| Random Forest | 0.6169 | 0.6104 | 0.6169 | 0.6134 |

El modelo seleccionado fue **Random Forest**, debido a que obtuvo el mejor Accuracy y F1-score.

## API FastAPI

La API permite enviar los datos de un estudiante y obtener una predicción.

### Endpoint

`POST /predict`

### Documentación

La documentación interactiva de la API está disponible mediante Swagger en:

`/docs`

## Ejemplo de respuesta

```json
{
  "prediccion": "Dropout",
  "probabilidades": {
    "Dropout": 0.82,
    "Enrolled": 0.06,
    "Graduate": 0.12
  }
}