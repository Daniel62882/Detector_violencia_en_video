from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import requests
import cv2
import numpy as np
from model_loader import loaded_model_vgg16, loaded_model_cnn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])

UPLOAD_FOLDER = 'uploads/videos'

def read_video(path):
    frames = []
    vidcap = cv2.VideoCapture(path)
    success, img = vidcap.read()

    while success:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(img, (224, 224)))
        success, img = vidcap.read()

    return frames

def conv_feature_image(frames):
    conv_features = loaded_model_vgg16.predict(np.array(frames))
    return np.array(conv_features)

def resize_zeros(img_features, max_frames):
    rows, cols = img_features.shape[:2]
    zero_matrix = np.zeros((max_frames - rows, cols))
    return np.concatenate((img_features, zero_matrix), axis=0)

def predict_via_HTTP(video_to_predict, model_name, model_version, port):
    frames = read_video(video_to_predict)
    img_features = conv_feature_image(frames)
    img_features_resized = resize_zeros(img_features, 190)

    # Model parameters and prediction via HTTP logic here
    test_image = image.array_to_img(img_features_resized[0])
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image.astype('float32')

    data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})
    headers = {"content-type": "application/json"}
    uri = f'http://127.0.0.1:{port}/v{model_version}/models/{model_name}:predict'

    json_response = requests.post(uri, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions'][0]

    return predictions

@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save video file
        contents = await file.read()
        video_path = f"{UPLOAD_FOLDER}/{file.filename}"
        with open(video_path, 'wb') as f:
            f.write(contents)

        # Model parameters
        model_name = 'violencia'  # Replace with your model name
        model_version = '1'  # Replace with your model version
        port_HTTP = '9501'  # Replace with the correct port

        # Make prediction
        predictions = predict_via_HTTP(video_path, model_name, model_version, port_HTTP)

        # Process results
        index = np.argmax(predictions)
        classes = [0,1]  # Adjust according to your model classes
        label = classes[index]
        score = predictions[index]

        # Build JSON response
        response = {
            "predictions": [{"label": label, "score": float(score)}]
        }

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)