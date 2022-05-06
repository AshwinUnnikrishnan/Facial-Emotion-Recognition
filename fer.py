# Created calibration.py by histravelstories on 6:47 PM under Project6

import cv2 as cv
from lib import detectFace, modelSelection
from modelsD import ResNet
import torch

import torchvision.transforms as tt

classLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

modelNumber = int(input("Enter the model to run \nResnet : 1\n"))


model = modelSelection(modelNumber, classLabels)

cam = cv.VideoCapture(1)

#To store face locations
face_locations = []

while True:
    flag = False
    retR, frame = cam.read()

    frame, cropped_image = detectFace(frame, model)

    operation = cv.waitKey(1)
    if operation == ord('q'):                   #quits the infinite loop
        break
    cv.imshow('VideoR', frame)
    for i in range(len(cropped_image)):
        cv.imshow('cropped'+str(i), cropped_image[i])
        roi_gray = cv.resize(cropped_image[i], (48, 48), interpolation=cv.INTER_AREA)
        cv.imshow('ReducedSize'+str(i), roi_gray)
        gray = cv.cvtColor(roi_gray, cv.COLOR_BGR2GRAY)
        cv.imshow('GrayScale'+str(i), gray)


cam.release()
cv.destroyAllWindows()