# Importar las bibliotecas necesarias
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import os

# Crear la aplicación Flask
app = Flask(__name__, static_url_path='')

# Ruta al modelo entrenado
modelo_path = r'C:\Users\USUARIO1-8.DESKTOP-CIJ8H1Q.000\Downloads\tourly-master\clasificador_imagenes\modelo_entrenado.h5'

# Cargar el modelo entrenado
model = load_model(modelo_path)

# Función para predecir la clase de una imagen
def predecir_clase_imagen(image_path, model):
    img = keras_image.load_img(image_path, target_size=(128, 128))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Obtener la clase predicha
    return predicted_class[0]

# Ruta para manejar la solicitud POST de clasificación de imágenes
@app.route('/classify', methods=['POST'])
def clasificar_imagen():
    try:
        # Obtener los datos JSON de la solicitud
        data = request.get_json()
        image_data = data['image']

        # Decodificar la imagen base64 y guardarla temporalmente
        temp_image_path = 'temp_image.jpg'  # Nombre temporal para la imagen
        with open(temp_image_path, 'wb') as f:
            f.write(image_data.encode('utf-8'))  # Decodificar correctamente la imagen base64

        # Predecir la clase de la imagen subida
        predicted_class = predecir_clase_imagen(temp_image_path, model)

        # Eliminar la imagen temporal
        os.remove(temp_image_path)

        # Devolver la clasificación como JSON
        return jsonify({'classification': str(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para servir la página HTML principal
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

# Iniciar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
