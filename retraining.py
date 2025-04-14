import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv("email_phishing_data.csv")

# Separar las características (X) y la variable objetivo (y)
X = data.drop("target", axis=1)  # Ajusta 'target' según el  archivo
y = data["target"]

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Obtener las probabilidades de predicción
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calcular las métricas ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"Métricas AUC: {roc_auc}")

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"Curva ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()
