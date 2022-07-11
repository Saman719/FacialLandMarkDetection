from importlib.resources import path
from keras.models import load_model
import data_handler as my_data_handler
import os
import cv2


def predictSamples():
    model = load_model('./Code/landmark-only.h5')
    filenames = os.listdir('./code/samples')
    paths = []
    for filename in filenames:
        paths.append('./Code/samples/' + filename)
    my_data_handler.predict(paths, model)


def predictLiveCam():
    model = load_model('./Code/landmark-only.h5')
    cap = cv2.VideoCapture(0)
    while True:
        ret, camImage = cap.read()
        my_data_handler.predict_live(camImage, model)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    predictSamples()
