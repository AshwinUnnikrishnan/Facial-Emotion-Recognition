import numpy as np
import torchvision.transforms as tt
import torch
import cv2 as cv

def predictEmotion(frame, model):
    classLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    roi_gray = cv.resize(frame, (48, 48), interpolation=cv.INTER_AREA)

    if np.sum([frame]) != 0:
        roi = tt.functional.to_pil_image(roi_gray)
        roi = tt.functional.to_grayscale(roi)
        roi = tt.ToTensor()(roi).unsqueeze(0)

        # make a prediction on the ROI
        tensor = model(roi)
        pred = torch.max(tensor, dim=1)[1].tolist()
        print(pred[0])
        label = classLabels[pred[0]]
        return label
