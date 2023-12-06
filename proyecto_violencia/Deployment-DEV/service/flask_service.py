# Importar las bibliotecas necesarias
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from model_loader import loadModelH5

# Args
import argparse

# Configurar los argumentos del programa
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, help="Número de puerto del servicio.")
args = vars(ap.parse_args())

# Puerto del servicio
port = args['port']
print("Puerto reconocido:", port)

# Parámetros
UPLOAD_FOLDER = 'uploads/videos'
ALLOWED_EXTENSIONS = set(['mp4'])

# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)

# Variables globales
# Carga el modelo utilizando la función importada
loaded_model = loadModelH5()

# Funciones
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_video(path):
    frames = []
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()

    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(image, (224, 224)))
        success, image = vidcap.read()

    return frames

def conv_feature_image(frames):
    max_frames = 190
    frames = frames[:max_frames]
    
    while len(frames) < max_frames:
        frames.append(np.zeros((224, 224, 3)))

    conv_features = [loaded_model.predict(np.expand_dims(frame, axis=0))[0] for frame in frames]

    return np.array(conv_features)

def resize_zeros(img_features, max_frames):
    rows, col = img_features.shape
    zero_matrix = np.zeros((max_frames - rows, col))
    return np.concatenate((img_features, zero_matrix), axis=0)

# Ruta para clasificar videos
@app.route('/model/predict/', methods=['POST'])
def predict_video():
    data = {"success": False}
    if request.method == "POST":
        if 'file' not in request.files:
            print('No se recibió el archivo')
            return jsonify(data)

        file = request.files['file']
        if file.filename == '':
            print('No se seleccionó ningún archivo')
            return jsonify(data)

        if file and allowed_file(file.filename):
            print("\nNombre de archivo recibido:", file.filename)
            filename = secure_filename(file.filename)
            tmpfile = ''.join([UPLOAD_FOLDER, '/', filename])
            file.save(tmpfile)
            print("\nNombre de archivo almacenado:", tmpfile)

            # Procesar video
            frames = read_video(tmpfile)
            img_features = conv_feature_image(frames)
            img_features = resize_zeros(img_features, 190)  # Asegúrate de que este número sea correcto

            # Predecir con el modelo cargado
            predictions = loaded_model.predict(np.array([img_features]))[0]
            class_pred = "violencia" if predictions >= 0.5 else "no-violencia"
            class_prob = float(predictions)

            print("Etiqueta de predicción:", class_pred)
            print("Probabilidad de predicción: {:.2%}".format(class_prob))

            # Resultados en formato Json
            data["predictions"] = [{"label": class_pred, "score": class_prob}]

            # Éxito
            data["success"] = True

    return jsonify(data)

# Resto de tu código Flask...

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, threaded=False)


