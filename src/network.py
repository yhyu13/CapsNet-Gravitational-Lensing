from tf_util import *
from util import *


class Network:


    def __init__():
        pass
    
    
    def init_IO():

        x = tf.placeholder(tf.float32, shape=[None, numpix_side*numpix_side])   #placeholder for input image
        y_label = tf.placeholder(tf.float32, shape=[None,num_out])    #placeholder for output parameters during training
        x_image0 = tf.reshape(x, [-1,numpix_side,numpix_side,1])

        # removing image intensity bias: filter image with a 4X4 filter and remove from image
        MASK = tf.abs(tf.sign(x_image0))
        XX =  x_image0 +  ( (1-MASK) * 1000.0)
        bias_measure_filt = tf.constant((1.0/16.0), shape=[4, 4, 1, 1])
        bias_measure = tf.nn.conv2d( XX , bias_measure_filt , strides=[1, 1, 1, 1], padding='VALID')
        im_bias = tf.reshape( tf.reduce_min(bias_measure,axis=[1,2,3]) , [-1,1,1,1] )
        x_image = x_image0 - (im_bias * MASK )
        
    def init_model():
        pass
        
    def init_summary():
        pass
        
    def init_optimizer():
        pass
        
    def init_cv2_display():
        pass
        
    def train_with_test(test=True):
        pass
    
    def test():
        pass
