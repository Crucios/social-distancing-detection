import numpy as np
import tensorflow as tf
import cv2
import time

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
    def __init__(self, model_path="models/best.pt"):
        """
        Initialization function
        @ model_path : path to the model 
        """
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, classes=1)
        checkpoint_ = torch.load(model_path)['model']
        model.load_state_dict(checkpoint_.state_dict())

        copy_attr(model, checkpoint_, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())

        self.model = model.fuse().autoshape()

    def predict(self,img):
        """
        Get the predicition results on 1 frame
        @ img : our img vector
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        img_exp = np.expand_dims(img, axis=0)
        # Pass the inputs and outputs to the session to get the results 
        (boxes, scores, classes) = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'), self.detection_graph.get_tensor_by_name('detection_scores:0'), self.detection_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'): img_exp})
        return (boxes, scores, classes)  

