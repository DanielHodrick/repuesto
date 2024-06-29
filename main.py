from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Proporciona la ruta correcta al archivo del modelo
model_path = 'C:/Users/USUARIO1-8.DESKTOP-CIJ8H1Q.000/Downloads/tourly-master/clasificador_imagenes/modelo_entrenado.h5'
model = load_model(model_path)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegúrate de que la carpeta de cargas existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def classify_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]
    classes = ['Clase 1', 'Clase 2', 'Clase 3']  # Cambia esto según tus clases
    return classes[class_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        classification = classify_image(file_path)
        return jsonify({'image_url': file_path, 'classification': classification})

if __name__ == '__main__':
    app.run(debug=True)
