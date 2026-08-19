import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Cargar datos
# ==========================================

df = pd.read_csv("data/data.csv", sep=";")


# ==========================================
# 2. Variables predictoras
# ==========================================

features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]


X = df[features]
y = df["Target"]


# ==========================================
# 3. Variables categóricas
# ==========================================

categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


# ==========================================
# 4. Variables numéricas
# ==========================================

numeric_features = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]


# ==========================================
# 5. Preprocesamiento
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)


# ==========================================
# 6. Modelo Random Forest
# ==========================================

modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 7. Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", modelo)
    ]
)


# ==========================================
# 8. División Holdout
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. Entrenamiento
# ==========================================

print("Entrenando Random Forest...")

pipeline.fit(X_train, y_train)


# ==========================================
# 10. Predicción
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 11. Evaluación
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("RESULTADOS DEL MODELO FINAL")
print("==========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")


print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("==========================================")
print("MATRIZ DE CONFUSIÓN")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Dropout", "Enrolled", "Graduate"]
)

print(cm)


# ==========================================
# 12. Guardar modelo
# ==========================================

joblib.dump(
    pipeline,
    "models/modelo.pkl"
)

print("\n==========================================")
print("MODELO GUARDADO")
print("==========================================")

print("Archivo: models/modelo.pkl")