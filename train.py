import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Cargar la base de datos
# ==========================================

df = pd.read_csv("data/data.csv", sep=";")


# ==========================================
# 2. Seleccionar variables predictoras
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


# ==========================================
# 3. Crear X e y
# ==========================================

X = df[features]
y = df["Target"]


# ==========================================
# 4. División Holdout 80/20
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. Crear modelo Random Forest
# ==========================================

modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 6. Entrenar modelo
# ==========================================

modelo.fit(X_train, y_train)


# ==========================================
# 7. Realizar predicciones
# ==========================================

y_pred = modelo.predict(X_test)


# ==========================================
# 8. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("==========================================")
print("RESULTADOS DEL MODELO")
print("==========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")


# ==========================================
# 9. Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. Matriz de confusión
# ==========================================

print("==========================================")
print("MATRIZ DE CONFUSIÓN")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Dropout", "Enrolled", "Graduate"]
)

print(cm)