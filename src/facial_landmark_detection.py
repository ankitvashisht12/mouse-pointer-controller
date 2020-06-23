'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from base_model import BaseModel

class Model_Facial_Landmark_Detection(BaseModel):
    '''
    Class for the Facial Landmark Detection Model.
    '''

    def preprocess_output(self, outputs, w, h, prob):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = []
        points = outputs["align_fc3"][0]
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
