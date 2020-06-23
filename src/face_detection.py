'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from base_model import BaseModel

class Model_Face_Detection(BaseModel):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs, w, h, prob):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = []
        for box in outputs["detection_out"][0][0]:
            if box[1] == 1 and box[2] > prob:
                outs.append([int(box[3]*w), int(box[4]*h), int(box[5]*w), int(box[6]*h)])

        return outs
