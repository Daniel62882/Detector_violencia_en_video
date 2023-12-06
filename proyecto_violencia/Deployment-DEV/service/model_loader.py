import tensorflow as tf
from tensorflow.keras.models import load_model

def loadModelH5(model_path, model_file):
    with tf.keras.utils.custom_object_scope({'KerasLayer': tf.keras.layers.Layer}):
        loaded_model = load_model(model_path + model_file)

    print(model_file, "Cargado desde el disco >> ", loaded_model)
    return loaded_model

# Cargar el modelo VGG16
MODEL_VGG16_FILE = "violencia_model_full_tf2.h5"  # Reemplaza con el nombre de tu modelo VGG16
MODEL_VGG16_PATH = "/home/dasniel298/models/model/tf3x/keras/full/"  # Reemplaza con la ruta a tu modelo VGG16
loaded_model_vgg16 = loadModelH5(MODEL_VGG16_PATH, MODEL_VGG16_FILE)

# Cargar el modelo CNN
MODEL_CNN_FILE = "violencia_model_full_tf2.h5"  # Reemplaza con el nombre de tu modelo CNN
MODEL_CNN_PATH = "/home/dasniel298/models/model/tf2x/keras/full/"  # Reemplaza con la ruta a tu modelo CNN
loaded_model_cnn = loadModelH5(MODEL_CNN_PATH, MODEL_CNN_FILE)
