from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ==========================================
# Crear aplicación FastAPI
# ==========================================

app = FastAPI(
    title="API de Predicción de Rendimiento Estudiantil",
    description="API para predecir el estado académico de estudiantes mediante Machine Learning",
    version="1.0.0"
)


# ==========================================
# Cargar modelo entrenado
# ==========================================

modelo = joblib.load("models/modelo.pkl")


# ==========================================
# Estructura de datos de entrada
# ==========================================

class Estudiante(BaseModel):

    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    previous_qualification_grade: float
    nacionality: int
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float


# ==========================================
# Endpoint principal
# ==========================================

@app.get("/")
def inicio():

    return {
        "mensaje": "API de predicción estudiantil funcionando"
    }


# ==========================================
# Endpoint de salud
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ==========================================
# Endpoint de predicción
# ==========================================

@app.post("/predict")
def predecir(estudiante: Estudiante):

    datos = {
        "Marital status": estudiante.marital_status,
        "Application mode": estudiante.application_mode,
        "Application order": estudiante.application_order,
        "Course": estudiante.course,
        "Daytime/evening attendance\t": estudiante.daytime_evening_attendance,
        "Previous qualification": estudiante.previous_qualification,
        "Previous qualification (grade)": estudiante.previous_qualification_grade,
        "Nacionality": estudiante.nacionality,
        "Mother's qualification": estudiante.mothers_qualification,
        "Father's qualification": estudiante.fathers_qualification,
        "Mother's occupation": estudiante.mothers_occupation,
        "Father's occupation": estudiante.fathers_occupation,
        "Admission grade": estudiante.admission_grade,
        "Displaced": estudiante.displaced,
        "Educational special needs": estudiante.educational_special_needs,
        "Debtor": estudiante.debtor,
        "Tuition fees up to date": estudiante.tuition_fees_up_to_date,
        "Gender": estudiante.gender,
        "Scholarship holder": estudiante.scholarship_holder,
        "Age at enrollment": estudiante.age_at_enrollment,
        "International": estudiante.international,
        "Unemployment rate": estudiante.unemployment_rate,
        "Inflation rate": estudiante.inflation_rate,
        "GDP": estudiante.gdp
    }

    # Convertir datos a DataFrame
    df = pd.DataFrame([datos])

    # Realizar predicción
    prediccion = modelo.predict(df)[0]

    # Obtener probabilidades
    probabilidades = modelo.predict_proba(df)[0]

    clases = modelo.classes_

    probabilidades_dict = {
        clase: round(float(probabilidad), 4)
        for clase, probabilidad in zip(clases, probabilidades)
    }

    return {
        "prediccion": prediccion,
        "probabilidades": probabilidades_dict
    }