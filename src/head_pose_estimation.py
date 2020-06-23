'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from base_model import BaseModel

class Model_Head_Pose_Estimation(BaseModel):
    '''
    Class for the Head Pose Estimation Model.
    '''
    
    def preprocess_output(self, outputs, w, h, prob):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        p = outputs["angle_p_fc"][0][0]
        r = outputs["angle_r_fc"][0][0]
        y = outputs["angle_y_fc"][0][0]
      
        return (p, r, y)
