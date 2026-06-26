import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración inicial de la interfaz
st.set_page_config(page_title="Clasificador Perros/Gatos IA", layout="centered")
st.title("Modelo predictivo Perros y Gatos - Clase de IA - Edy Martinez")
st.write("Suba una imagen para clasificar con el modelo MobileNetV2 entrenado localmente")

# Dimensiones estándar requeridas por MobileNetV2
IMG_SIZE = (224, 224)

# Configuración de las rutas locales según tus archivos generados
MODEL_PATH = Path("modelo_perros_gatos_mobilenetv2.h5")
CLASS_PATH = Path("clases.txt")

# Diccionario de traducción para las etiquetas de tus carpetas
LABELS_ES = {
    "gatos": "Gato 🐱",
    "perros": "Perro 🐶"
}

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        # Cargamos el archivo .h5 generado en el entrenamiento
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.error("No se encontró el modelo. Asegúrate de tener 'modelo_perros_gatos_mobilenetv2.h5' en la misma carpeta que app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        # Leer el archivo clases.txt línea por línea
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    # Clases por defecto si el archivo no existiera
    return ["gatos", "perros"]

def preparar_imagen(img):
    # Convertir a RGB por si tiene transparencias (PNG) y redimensionar
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Escalado de píxeles dividiendo entre 255.0, idéntico al ImageDataGenerator del entrenamiento
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    
    # Ordenar las predicciones para mostrar los resultados en pantalla
    # Como solo tenemos 2 clases, las ordenamos de mayor a menor probabilidad
    top_indices = np.argsort(preds)[::-1]
    
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top_indices
    ]

# Cargar el modelo y las clases en memoria utilizando la caché de Streamlit
modelo = cargar_modelo()
clases = cargar_clases()

# Componente interactivo para cargar archivos en la web
archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    # Cargar y desplegar la imagen en la interfaz gráfica
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    # Realizar el análisis con la red neuronal
    resultados = predecir(imagen)
    
    st.subheader("Resultado")
    # Mostrar la predicción ganadora con un recuadro verde de éxito
    st.success(f"Predicción principal: {resultados[0][0]} ({resultados[0][1]:.2f}%)")

    # Mostrar la tabla comparativa de probabilidades de la IA
    st.write("Probabilidades del modelo:")
    for clase, prob in resultados:
        st.write(f"**{clase}**: {prob:.2f}%")
else:
    st.info("Cargue una imagen de un perro o un gato para iniciar la clasificación.")
