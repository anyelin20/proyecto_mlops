# Proyecto MLOps

## Integrantes del Equipo
- Esteban Ramirez (Data Scientist)
- Allan Montes (MLOps Engineer)
- Anyelin Arias (Software Engineer)

## ✅ Checklist del Proyecto

- [x] El código de un modelo de AI para resolver un problema de machine learning o deep learning en un notebook de Python `.ipynb` o bien, la herramienta o lenguaje de su preferencia de su preferencia.
- [x] El script, software o código en general para automatizar el reentrenamiento del modelo.
- [x] El versionamiento del código que contenga el código realizado para el modelo de AI y el Script de reentrenamiento. Tomar en cuenta que en este versionamiento debe trabajar todo el equipo haciendo los pull requests necesarios en el branch de `develop`. También debe contener al menos el branch de `develop`, `staging` y `main` (o `master` como deseen trabajarlo).
- [x] Realización de un API para consumir el modelo de inteligencia artificial o bien, una interface gráfica web con Streamlit, ReactJS o el framework para GUIs web de su preferencia.
- [x] Realizar el sistema de conteinerización con Docker o de su preferencia, listo para el deployment.
- [x] Realizar la seguridad en la validación de entradas del API y análisis estático del código usando DeepSource o su herramienta de preferencia
- [x] Debe realizar el CI/CD en Github Actions o su herramienta de preferencia (Jenkins, CircleCI, Gitlab, etc...) para el reentrenamiento del modelo y también para enviarlo al ambiente de desarrollo en un servidor en la nube de su preferencia (puede ser AWS EC2 o cualquier otro).
- [x] Poner las instrucciones de ejecución y documentación del modelo de AI y el API (en caso de interface gráfica también se necesitan saber cuáles son los inputs y los outputs del modelo) en un `README.md` dentro del repositorio central.
- [x] Realizar el versionamiento del modelo y de los datos usando DVC o su herramienta de preferencia y los datos junto con el modelo deben estar subidos a un object storage de la nube de su preferencia (puede ser AWS S3).

## Descripción del problema
Utilizando características particulares de corres electrónicos de phising y de los que no son phising,
se analizan con el fin de poder detectar posibles correos maliciosos.

## Dataset
Los datos utilizados para entrenar el modelo provienen de Kaggle:
[Email Phishing Dataset](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset?resource=download)

### Inputs
El dataset contiene características extraídas de emails, como:

- `num_words:` Número de palabras en el email
- `num_unique_words:` Número de palabras únicas
- `num_stopwords:` Cantidad de palabras comunes (stopwords)
- `num_links:` Número de enlaces
- `num_unique_domains:` Cantidad de dominios únicos
- `num_email_addresses:` Número de direcciones de email
- `num_spelling_errors:` Cantidad de errores ortográficos
- `num_urgent_keywords:` Número de palabras que indican urgencia

### Output

- `label:` Etiqueta binaria (0: legítimo, 1: phishing)

## Modelo de Machine Learning

Para este proyecto se utilizó un **Random Forest Classifier**, optimizado mediante Grid Search para encontrar los mejores hiperparámetros.

## Estructura del Repositorio

```
📁 proyecto_mlops/                # Carpeta raíz del proyecto
│
├── 📁 .github/workflows/         # Automatizaciones CI/CD
│   └── 📄 main.yml               # Pipeline de GitHub Actions
│
├── 📁 data/                      # Dataset versionado con DVC
│   └── 📄 mini_email_phishing_data.csv.dvc
│
├── 📁 notebooks/                 # Notebooks y artefactos de entrenamiento
│   ├── 📓 classification_model.ipynb      # Desarrollo del modelo
│   ├── 📦 classification_model.joblib     # Modelo serializado
│   ├── 📊 confusion_matrix.png            # Métricas visuales
│   ├── 📊 feature_importance.png
│   └── 📊 roc_curve.png
│
├── 📄 Dockerfile                 # Imagen de producción para la API
├── 📄 download_data.py           # Descarga del dataset desde S3
├── 📄 endpoints.py               # FastAPI con endpoint de inferencia
├── 📄 retraining.py              # Script de re-entrenamiento y métricas
├── 📄 requirements.txt           # Dependencias del proyecto
├── 📄 .gitignore                 # Archivos/dirs a excluir del control de versiones
└── 📄 README.md                  # Documentación del proyecto
```

## Requerimientos

Necesita instalar las siguientes herramientas:
- Python 3.12.10 o superior
- pip
- Virtualenv (opcional)

Comando para instalar las dependencias del proyecto:
```bash
pip install -r requirements.txt

