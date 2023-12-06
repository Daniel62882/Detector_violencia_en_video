from flask import Flask, request, jsonify, render_template
import numpy as np
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
    # Simulando una predicción
    return np.random.rand(len(frames))

def resize_zeros(img_features, max_frames):
    rows, cols = img_features.shape[:2]
    zero_matrix = np.zeros((max_frames - rows, cols, 3))
    return np.concatenate((img_features, zero_matrix), axis=0)

def format_result(prediction):
    # Formatear el resultado con un estilo
    return f"<table style='border: 2px solid #1c87c9; border-radius: 10px; padding: 10px; margin: 20px auto; max-width: 400px;'><tr><td>Prediction:</td><td style='color: {'red' if prediction >= 0.5 else 'green'};'>{prediction}</td></tr></table>"

@app.route('/model/predict/', methods=['POST'])
def predict_video():
    try:
        # Obtener el archivo de video desde la solicitud
        video_file = request.files['file']
        video_path = "uploads/videos/" + video_file.filename
        video_file.save(video_path)

        # Procesar el video (simulando una predicción)
        frames = read_video(video_path)
        img_features = conv_feature_image(frames)
        img_features_resized = resize_zeros(img_features, 190)

        # Hacer la predicción (simulando una predicción)
        prediction = np.random.rand()

        # Crear respuesta HTML con estilo
        result_html = format_result(prediction)

        return result_html

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
