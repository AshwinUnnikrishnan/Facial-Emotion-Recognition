import numpy as np
import face_recognition
import cv2 as cv
from predict import predictEmotion
import torch
from modelsD import ResNet

def detectFace(frame, model):
    # makes things faster
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # face_recognition only works with rgb but our input is bgr
    rgb_small_frame = small_frame[:, :, ::-1]

    # get the locations in the list, there might be multiple faces too
    face_locations = face_recognition.face_locations(rgb_small_frame)

    cropped_image = []
    for (top, right, bottom, left) in face_locations:
        # scaling back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # drawing bounding box on the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # this image should be the input to our model
        temp = frame[top:bottom, left:right]
        cropped_image.append(temp)
        emotion = predictEmotion(temp, model)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, cropped_image

def modelSelection(modelSelection, class_labels):
    if modelSelection == 1:
        model_state = torch.load("model/resNet.pth", map_location='cpu')
        model = ResNet(1, len(class_labels))
        model.load_state_dict(model_state)
    return model