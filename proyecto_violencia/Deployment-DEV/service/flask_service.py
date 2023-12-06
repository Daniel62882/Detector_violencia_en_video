from flask import Flask, request, render_template
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
    rows, cols = img_features.shape[:2]
    zero_matrix = np.zeros((max_frames - rows, cols, 3))
    return np.concatenate((img_features, zero_matrix), axis=0)

@app.route('/model/predict/', methods=['POST'])
def predict_video():
    try:
        video_file = request.files['file']
        video_path = "uploads/videos/" + video_file.filename
        video_file.save(video_path)

        frames = read_video(video_path)
        img_features = conv_feature_image(frames)
        img_features_resized = resize_zeros(img_features, 190)

        prediction = loaded_model_cnn.predict(np.array([img_features_resized]))

        threshold = 0.5

        result_html = """
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 20px;
                }
                .result-box {
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                .violent {
                    color: #FF0000;
                }
                .non-violent {
                    color: #00FF00;
                }
            </style>
        </head>
        <body>
            <div class="result-box">
                <h2>Resultado de la predicción</h2>
                <p>Confianza: {:.2%}</p>
                <p class="{}">{}</p>
            </div>
        </body>
        </html>
        """.format(prediction[0], 'violent' if prediction[0] >= threshold else 'non-violent', 'El video es violento' if prediction[0] >= threshold else 'El video no es violento')

        return result_html

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
