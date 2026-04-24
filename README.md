en la terminal del vCode ejecutar el comando:

python -m venv venv

luego activar el entorno virtual con el comando:

venv\Scripts\Activate.ps1

instalar las librerías necesarias con el comando:

pip install pandas numpy matplotlib scikit-learn imbalanced-learn streamlit fastapi uvicorn joblib

guardar las dependencias en un archivo requirements.txt con el comando:

pip freeze > requirements.txt

incluir el dataset descargado creando una carpeta data con el nombre de data


por ultimo crear una carpeta reports


para ejecutar el programa usar el comando

streamlit run app/frontend/dashboard.py
