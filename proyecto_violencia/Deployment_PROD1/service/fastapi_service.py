from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import numpy as np
import cv2
import requests
import json
from model_loader import loaded_model_vgg16, loaded_model_cnn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])

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

def convert_frames_to_json(frames):
    return json.dumps({"signature_name": "serving_default", "instances": frames.tolist()})

def resize_zeros(img_features, max_frames):
    rows, cols = img_features.shape[:2]
    zero_matrix = np.zeros((max_frames - rows, cols))
    return np.concatenate((img_features, zero_matrix), axis=0)

@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Obtener el archivo de video desde la solicitud
        contents = await file.read()

        # Guardar el archivo en disco
        video_path = "uploads/videos/" + file.filename
        with open(video_path, 'wb') as f:
            f.write(contents)

        # Procesar el video
        frames = read_video(video_path)
        img_features = conv_feature_image(frames)
        img_features_resized = resize_zeros(img_features, 190)

        # Convertir frames a formato JSON
        frames_json = convert_frames_to_json(np.array([img_features_resized]))

        # Enviar solicitud a TensorFlow Serving
        model_name = 'violencia'  # Reemplaza con el nombre de tu modelo
        model_version = '1'  # Reemplaza con la versión de tu modelo
        port = '8501'  # Reemplaza con el puerto de tu servidor TensorFlow Serving

        uri = f'http://127.0.0.1:{port}/v{model_version}/models/{model_name}:predict'
        headers = {"content-type": "application/json"}
        response = requests.post(uri, data=frames_json, headers=headers)

        # Obtener la predicción del resultado
        predictions = response.json().get('predictions', None)

        if predictions is not None:
            # Obtener la probabilidad de violencia
            probability_of_violence = predictions[0] if isinstance(predictions, list) else predictions

            # Definir umbral de decisión
            threshold = 0.5

            # Clasificar la predicción
            label = "violencia" if probability_of_violence >= threshold else "no_violento"
            score = float(probability_of_violence)

            response_data = {"prediction": score, "label": label}
            return JSONResponse(content=response_data, status_code=200)
        else:
            raise ValueError("La respuesta de TensorFlow Serving no contiene 'predictions'.")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
