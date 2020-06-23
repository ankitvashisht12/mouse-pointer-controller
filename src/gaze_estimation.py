'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
import cv2
from base_model import BaseModel

class Model_Gaze_Estimation(BaseModel):
    '''
    Class for the Gaze Estimation Model.
    '''
    def predict(self, left_eye, right_eye, head_pose_outs):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
    

        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        inputs = {"head_pose_angles" : head_pose_outs, "left_eye_image" : left_eye, "right_eye_image" : right_eye}
        
        start_inf = time.time()
        res = self.exec_net.infer(inputs)
        diff_inf = time.time() - start_inf
        return res['gaze_vector'][0], diff_inf
        


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        img = cv2.resize(image.copy(), (60, 60))
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape)

        return img


