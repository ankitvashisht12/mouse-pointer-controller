'''
This is a base class for a model.
'''

import os
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class BaseModel:
    '''
    Class for the Face Detection Model.
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
        self.model = IENetwork(self.model_structure, self.model_weights)

    

        # Add Extension if provided
        if self.extensions and self.device=="CPU":
            self.core.add_extension(self.extensions, self.device)

        # Check Model   
        if not self.check_model():
            print("Network Error while checking model")
            exit(1)
        
        # Load network
        self.exec_net = self.core.load_network(self.model, self.device)

        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))


    def predict(self, image, prob=0.5):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        w = image.shape[1]
        h = image.shape[0]

        img = self.preprocess_input(image)
        start_inf = time.time()
        res = self.exec_net.infer({self.input_blob: img})
        diff_inf = time.time() - start_inf
        out = self.preprocess_output(res, w , h, prob)
        return out, diff_inf

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
        pass
