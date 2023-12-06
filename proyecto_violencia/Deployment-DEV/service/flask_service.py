from flask import Flask, request, jsonify
import numpy as np
from model_loader import loaded_model_vgg16, loaded_model_cnn
import cv2

app = Flask(__name__)

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
    conv_features = loaded_model_vgg16.predict(np.array(frames))
    return np.array(conv_features)

def resize_zeros(img_features, max_frames):
    rows, cols = img_features.shape[:2]  # Obtener solo las dos primeras dimensiones
    zero_matrix = np.zeros((max_frames - rows, cols, 3))  # Asegurarse de que img_features tiene 3 dimensiones
    return np.concatenate((img_features, zero_matrix), axis=0)

@app.route('/model/predict/', methods=['POST'])
def predict_video():
    try:
        # Obtener el archivo de video desde la solicitud
        video_file = request.files['file']
        video_path = "uploads/videos/" + video_file.filename
        video_file.save(video_path)

        # Procesar el video
        frames = read_video(video_path)
        img_features = conv_feature_image(frames)
        img_features_resized = resize_zeros(img_features, 190)

        # Hacer la predicción
        prediction = loaded_model_cnn.predict(np.array([img_features_resized]))

        # Definir umbral de decisión
        threshold = 0.5

        # Mostrar resultado en la consola
        if prediction[0] >= threshold:
            print("El video es un situacion de violencia")
        else:
            print("El video no es violento")

        # Crear respuesta JSON
        result = {"prediction": float(prediction[0])}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)