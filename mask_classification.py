import pandas as pd
import matplotlib.pyplot as pyplot
import os
import numpy as np
from skimage import color
from skimage import io
import cv2
import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix
import time


path = '../mask/dataset/'
#data set = https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset

def prepare_data(dir, className):
    preparedData = []
    labels = []
    imageFiles = os.listdir(dir)
    for imageFile in imageFiles:
        image = preProcessImage(dir+imageFile)
        # pyplot.imshow(image, cmap='gray')
        preparedData.append(image)
        labels.append(className)
    return (preparedData, labels)

def  scaleImageAndFlatten(image):
    return np.array(cv2.resize(image, (200, 200))).flatten()

def convertToGrayscale(image):
    return scaleImageAndFlatten(color.rgb2gray(image))

def preProcessImage(imageFile):
    image = pyplot.imread(imageFile)
    return getConvertedImage(image)

def getConvertedImage(image):
    image = convertToGrayscale(image)
    return image

def createModel():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(128, activation=tf.nn.relu))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

def shuffle_data(list1, list2):
    np1 = np.array(list1)
    np2 = np.array(list2)
    indices = np.arange(np1.shape[0])
    np.random.shuffle(indices)
    np1 = np1[indices]
    np2 = np2[indices]
    return (np1, np2)

data, classes = prepare_data(path+"Train/Mask/", 1)
data2, classes2 = prepare_data(path+"Train/Non_Mask/", 0)

data.extend(data2)
classes.extend(classes2)

x, y = shuffle_data(data, classes)

model = createModel()

history = model.fit(x=x, y=y, epochs=15)
pyplot.plot(history.history['loss'])
pyplot.show()

def testData():
    global model
    data, classes = prepare_data(path+"Test/Mask/", 1)
    data2, classes2 = prepare_data(path+"Test/Non_Mask/", 0)
    data.extend(data2)
    classes.extend(classes2)
    data, classes = shuffle_data(data, classes)
    predictions = model.predict(data)
    y_pred = []
    for pred in predictions:
        y_pred.append(pred.argmax())
    y_pred = np.array(y_pred).reshape(-1)
    print(getAccuracy(y_pred, classes))
    print(cmatrix(y_pred, classes))

def getAccuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

def cmatrix(y_pred, y_true):
    labels = [1, 0]
    cm = confusion_matrix(y_true, y_pred, labels)
    pyplot.matshow(cm)
    return cm

def randomTestData(data, predictions):
    index = random.randrange(0, len(data))
    image = np.reshape(data[index], (200, 200))
    prediction = predictions[index].argmax()
    pyplot.title("Masked" if prediction == 1 else "Non Masked")
    pyplot.imshow(image)
frameData = None
newPred = []
dataList = []
def testLive():
    global frameData
    global newPred
    global dataList
    cap = cv2.VideoCapture(0)
    timeout = time.time() + 40
    count = 0
    while time.time() < timeout:
        ret, frame = cap.read()
        count = count+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameData = frame
        img = convertToGrayscale(frameData)
        dataList.append(img)
        newPred = model.predict(np.array(dataList))
        lastIndex = len(dataList) - 1
        cv2.putText(gray, 'Masked' if newPred[lastIndex].argmax() == 1 else "Non Masked", (10, 200), cv2.FONT_ITALIC, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(gray, str(count), (10, 200), cv2.FONT_ITALIC, 2,
        #             (255, 255, 255), 2, cv2.LINE_AA)
        print(newPred[lastIndex].argmax())
        cv2.imshow('frame', gray)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()