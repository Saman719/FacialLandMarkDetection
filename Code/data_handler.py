from importlib.resources import path
from struct import pack
import scipy.io
import os
import cv2
import numpy as np
import random
from sklearn.preprocessing import normalize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


WIDTH = 75
HEIGHT = 75
TRAIN_RATIO = 0.75


def train_test():
    data = []
    lbl = []
    packages = []
    file1 = open('./Code/Data/WiderSelected/annotations.txt', 'r')
    lines = file1.readlines()
    data_desc = []
    for line in lines:
        if len(line) > 2 and (line[1] == '-' or line[2] == '-' or (lines.index(line) == len(lines) - 1)):
            if(lines.index(line) == len(lines) - 1):
                data_desc.append(line)
            packages.append(data_desc[:])
            data_desc.clear()
        data_desc.append(line)
    packages.remove([])
    packages = random.sample(packages, len(packages))
    for package in packages:
        fullImage = cv2.imread("./Code/Data/WiderSelected/train/" +
                               package[0][:len(package[0])-1])
        # cv2.imshow('Facial Landmark After Resized', fullImage)
        # cv2.waitKey(0)
        for i in range(int(package[1])):
            points = package[i+2].split()
            points = [int(point) for point in points]
            faceImage = fullImage[points[1]: points[1] +
                                  points[3], points[0]:points[0] + points[2], ::]
            y_ratio = HEIGHT / faceImage.shape[0]
            x_ratio = WIDTH / faceImage.shape[1]
            img = cv2.resize(faceImage, (HEIGHT, WIDTH), fx=x_ratio,
                             fy=y_ratio, interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('Image After Resized', img)
            # cv2.waitKey(0)
            transformedPoints = calculateTrainLable(points[0], points[1],
                                                    x_ratio, y_ratio, points[4:])
            for i in range((len(transformedPoints) // 2)):
                img = cv2.circle(
                    img, (round(transformedPoints[2*i]), round(transformedPoints[2*i+1])), 2, (255, 0, 0), 1)
            # cv2.imshow('Facial Landmark After Resized', img)
            # cv2.waitKey(0)
            data.append(img)
            lbl.append(transformedPoints)
    return (data[:int(len(data)*TRAIN_RATIO)], lbl[:int(len(data)*TRAIN_RATIO)],
            data[int(len(data)*TRAIN_RATIO):], lbl[int(len(data)*TRAIN_RATIO):])


def calculateTrainLable(x, y, x_ratio, y_ratio, points):
    train_lbl = []
    for counter, point in enumerate(points):
        if counter % 2 == 0:
            train_lbl.append(
                (point - x) * x_ratio if (point - x) * x_ratio > 0 else 0)
        else:
            train_lbl.append(
                (point - y) * y_ratio if (point - y) * y_ratio > 0 else 0)
    return train_lbl


def predict(paths, model):
    images = []
    for path in paths:
        img = cv2.imread(path)
        y_ratio = HEIGHT / img.shape[0]
        x_ratio = WIDTH / img.shape[1]
        img = cv2.resize(img, (HEIGHT, WIDTH), fx=x_ratio,
                         fy=y_ratio, interpolation=cv2.INTER_CUBIC)
        normalizedImage = np.array(img, dtype=np.float64) / 255
        images = [normalizedImage]
        images = np.array(images, dtype=np.float64)
        prediction = model.predict(images)
        realImage = cv2.imread(path)
        prediction = prediction[0]
        for i in range((len(prediction) // 2)):
            img = cv2.circle(
                img, (round(prediction[2*i]), round(prediction[2*i+1])), 1, (255, 0, 0), 3)
            realImage = cv2.circle(
                realImage, (round(prediction[2*i] / x_ratio), round(prediction[2*i+1] / y_ratio)), 1, (255, 0, 0), 3)
        cv2.imshow("landmark after resize", img)
        cv2.imshow("landmark full size", realImage)
        cv2.waitKey(0)
        print(prediction)
    return images


def predict_live(realImage, model):
    img = realImage[:, :, :]
    y_ratio = HEIGHT / img.shape[0]
    x_ratio = WIDTH / img.shape[1]
    img = cv2.resize(img, (HEIGHT, WIDTH), fx=x_ratio,
                     fy=y_ratio, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float64) / 255
    images = [img]
    images = np.array(images, dtype=np.float64)
    prediction = model.predict(images)
    xPrediction, yPrediction, widthPrediction, heightPrediction = prediction[0]
    realX = int(xPrediction / x_ratio)
    realY = int(yPrediction / y_ratio)
    realWidth = int(widthPrediction / x_ratio)
    realHeight = int(heightPrediction / y_ratio)
    cv2.rectangle(realImage, (realX, realY), (realX+realWidth,
                                              realY+realHeight), (255, 0, 0), 4)
    cv2.imshow("Live Cam Prediction", realImage)
