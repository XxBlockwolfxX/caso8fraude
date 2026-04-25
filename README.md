<<<<<<< HEAD
en la terminal ejecutar el comando:
=======
en la terminal del vCode ejecutar el comando:
>>>>>>> 9aba5e6f79a86c76dd8bcf8750187d97bc25e2c4

python -m venv venv

luego activar el entorno virtual con el comando:

venv\Scripts\Activate.ps1

instalar las librerías necesarias con el comando:

pip install pandas numpy matplotlib scikit-learn imbalanced-learn streamlit fastapi uvicorn joblib

guardar las dependencias en un archivo requirements.txt con el comando:

pip freeze > requirements.txt

incluir el dataset descargado creando una carpeta data con el nombre de data


<<<<<<< HEAD
por ultimo crear una carpeta reports
=======
por ultimo crear una carpeta reports


para ejecutar el programa usar el comando, verificar de que estas en la carpeta raiz  ej: C:\Users\Admin\Desktop\caso8fraude

streamlit run app/frontend/dashboard.py
>>>>>>> 9aba5e6f79a86c76dd8bcf8750187d97bc25e2c4
