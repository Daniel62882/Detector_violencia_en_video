# Import Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import cv2
from model_loader import loadModelH5

# Args
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, help="Service PORT number is required.")
args = vars(ap.parse_args())

# Service port
port = args['port']
print("Port recognized: ", port)

# Params
UPLOAD_FOLDER = 'uploads/videos'
ALLOWED_EXTENSIONS = set(['mp4'])

# Initialize the application service (FLASK)
app = Flask(__name__)
CORS(app)

# Vars
global loaded_model
loaded_model = loadModelH5()

# Functions
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

def process_video(video_path):
    frames = read_video(video_path)
    img_features = conv_feature_image(frames)
    img_features = resize_zeros(img_features, max_frames)
    return img_features

def conv_feature_image(frames):
    conv_features = loaded_model.predict(np.array(frames))
    return np.array(conv_features)

def resize_zeros(img_features, max_frames):
    rows, col = img_features.shape
    zero_matrix = np.zeros((max_frames - rows, col))
    return np.concatenate((img_features, zero_matrix), axis=0)

# Ruta para clasificar videos
@app.route('/video/predict/', methods=['POST'])
def predict_video():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return jsonify(data)

        file = request.files['file']
        # if user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            print('No selected file')
            return jsonify(data)

        if file and allowed_file(file.filename):
            print("\nFilename received:", file.filename)
            filename = secure_filename(file.filename)
            tmpfile = ''.join([UPLOAD_FOLDER, '/', filename])
            file.save(tmpfile)
            print("\nFilename stored:", tmpfile)

            # processing video
            video_features = process_video(tmpfile)

            # predicting using the loaded model
            predictions = loaded_model.predict(np.array([video_features]))[0]
            class_pred = "violence" if predictions >= 0.5 else "non-violence"
            class_prob = float(predictions)

            print("Prediction Label:", class_pred)
            print("Prediction Prob: {:.2%}".format(class_prob))

            # Results as Json
            data["predictions"] = [{"label": class_pred, "score": class_prob}]

            # Success
            data["success"] = True

    return jsonify(data)

# Run the application
app.run(host='0.0.0.0', port=port, threaded=False)
