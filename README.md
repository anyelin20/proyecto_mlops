# Proyecto MLOps

## Integrantes del Equipo
- Esteban Ramirez
- Allan Montes
- Anyelin Arias

## ✅ Checklist del Proyecto

- [x] El código de un modelo de AI para resolver un problema de machine learning o deep learning en un notebook de Python `.ipynb` o bien, la herramienta o lenguaje de su preferencia de su preferencia.
- [ ] El script, software o código en general para automatizar el reentrenamiento del modelo.
- [ ] El versionamiento del código que contenga el código realizado para el modelo de AI y el Script de reentrenamiento. Tomar en cuenta que en este versionamiento debe trabajar todo el equipo haciendo los pull requests necesarios en el branch de `develop`. También debe contener al menos el branch de `develop`, `staging` y `main` (o `master` como deseen trabajarlo).
- [ ] Realización de un API para consumir el modelo de inteligencia artificial o bien, una interface gráfica web con Streamlit, ReactJS o el framework para GUIs web de su preferencia.
- [ ] Realizar el sistema de conteinerización con Docker o de su preferencia, listo para el deployment.
- [ ] Realizar la seguridad en la validación de entradas del API y análisis estático del código usando DeepSource o su herramienta de preferencia.
- [ ] Debe realizar el CI/CD en Github Actions o su herramienta de preferencia (Jenkins, CircleCI, Gitlab, etc...) para el reentrenamiento del modelo y también para enviarlo al ambiente de desarrollo en un servidor en la nube de su preferencia (puede ser AWS EC2 o cualquier otro).
- [ ] Poner las instrucciones de ejecución y documentación del modelo de AI y el API (en caso de interface gráfica también se necesitan saber cuáles son los inputs y los outputs del modelo) en un `README.md` dentro del repositorio central.
- [ ] Realizar el versionamiento del modelo y de los datos usando DVC o su herramienta de preferencia y los datos junto con el modelo deben estar subidos a un object storage de la nube de su preferencia (puede ser AWS S3).


## Requerimientos

Necesita instalar las siguientes herramientas:
- Python 3.12.10 o superior
- pip
- Virtualenv (opcional)

Comando para instalar las dependencias del proyecto:
```bash
pip install -r requirements.txt

