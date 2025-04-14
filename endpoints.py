# api_modelo.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo guardado
modelo = joblib.load("notebooks/classification_model.joblib")

# Crear la app de FastAPI
app = FastAPI()

class DatosEntrada(BaseModel):
    num_words: float
    num_unique_words: float
    num_stopwords: float
    num_links: float
    num_unique_domains: float
    num_email_addresses: float
    num_spelling_errors: float
    num_urgent_keywords: float

    

# Ruta para hacer predicciones
@app.post("/predecir")
def predecir(data: DatosEntrada):
    entrada = np.array([[
    data.num_words, data.num_unique_words, data.num_stopwords,
    data.num_links, data.num_unique_domains, data.num_email_addresses,
    data.num_spelling_errors, data.num_urgent_keywords
]])

    prediccion = modelo.predict(entrada)[0]
    return {"prediccion": int(prediccion)}
