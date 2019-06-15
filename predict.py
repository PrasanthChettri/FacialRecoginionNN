import cv2
from model import NN
import numpy as np
import torch

cap = cv2.VideoCapture(0)
i = 0 
classify = 1 
labels =[] 
Model = NN(batch_size = 1)
Model.load_state_dict(torch.load("1"))
Model.eval()
tardict = {1 : 'Face Detected' , 0 : 'Undetected'  }

while True:
    i += 1
    ret  , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    cv2.imshow('feed' , frame)
    gray = torch.from_numpy(gray).view(1 , 1, 480 , 640).float()
    output = torch.round(Model.forward(gray))
    output = output.item()
    print (tardict[output])
    if output != 0:
        input()
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break 
   
