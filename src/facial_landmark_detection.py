'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model_Facial_Landmark_Detection:
    '''
    Class for the Facial Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0] + '.bin'
        self.device = device
        self.extensions = extensions
        self.core = None
        self.model = None
        self.exec_net = None
        self.input_blob = None
        self.output_blob = None
        self.infer_request = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        
        self.core = IECore()
        self.model = self.core.read_network(self.model_structure, self.model_weights)

        # Add Extension if provided
        if self.extensions and self.device=="CPU":
            self.core.add_extension(self.extensions, self.device)

        # Check Model   
        if not self.check_model():
            print("Network Error")
            exit(0)
        
        # Load network
        self.exec_net = self.core.load_network(self.model, self.device)

        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        w = image.shape[1]
        h = image.shape[0]

        img = self.preprocess_input(image)
        res = self.exec_net.infer({self.input_blob: img})
        
        out = self.preprocess_output(res["align_fc3"], w , h)
        return out

    def check_model(self):
        supported_layers = self.core.query_network(self.model, self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            return False
        
        return True

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.model.inputs[self.input_blob].shape
        img = cv2.resize(image.copy(), (net_input_shape[3], net_input_shape[2]))
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape)

        return img

    def preprocess_output(self, outputs, w, h):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = []
        points = outputs[0]
        left_eye_x0 = int(points[0]*w)
        left_eye_y0 = int(points[1]*h)
        left_eye_x1 = int(points[2]*w)
        left_eye_y1 = int(points[3]*h)

        right_eye_x0 = int(points[4]*w)
        right_eye_y0 = int(points[5]*h)
        right_eye_x1 = int(points[6]*w)
        right_eye_y1 = int(points[7]*h)
        
        outs.append([(left_eye_x0, left_eye_y0), (left_eye_x1, left_eye_y1), (right_eye_x0, right_eye_y0), (right_eye_x1, right_eye_y1)])

        return outs
