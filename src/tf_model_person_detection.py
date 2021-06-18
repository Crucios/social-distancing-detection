import numpy as np
import cv2
import time
import torch

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class Model:
    """
    Class that contains the model and all its functions
    """
    def __init__(self, model_path="../models/best_10.pt"):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)


    def predict(self,img):
        """
        Get the predicition results on 1 frame
        @ img : our img vector
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # img_exp = np.expand_dims(img, axis=0)
        # Pass the inputs and outputs to the session to get the results 
        results = self.model(img)
        boxes = results.xyxy[0][:,:4].numpy()
        scores = results.xyxy[0][:,4].numpy()
        print(boxes)
        print(scores)
        return (boxes, scores)  
    
    def predict_real(self,img):
        return self.model(img)

