from config import *
from tf_util import cost_tensor, init_xy_placeholder
from util import *

from .model import *
import cv2

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

class Network:

    def __init__(self, hps, FLAGS):
        self.hps = hps
        self.FLAGS = FLAGS
        
        tf.reset_default_graph()
        g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.33
        self.sess = tf.Session(config=config, graph=g)
        
        with g.as_default():
            self.init_preprocess()
            self.init_model()
            self.init_summary()
            self.init_var()
    
    def init_preprocess():
        self.x, self.y_label, self.x_image = init_xy_placeholder()
        
    def init_model():
        models = {'cnn': lambda: CNNBaseline(self.hps, self.x_image, self.y_label),
                  'cap': lambda: CapsNet(self.hps, self.x_image, self.y_label)}
        self.model = models[self.FLAGS.model]()
        logger.debug("Building Model...")
        self.model.build_graph()
        
    def init_summary():
        self.summary = self.model.summaries
        self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./test_log")
        
        var_to_save = tf.trainable_variables() + [var for var in tf.global_variables()
                                                      if ('bn' in var.name) and ('Adam' not in var.name) and ('Momentum' not in var.name) or ('global_step' in var.name)]
        logger.debug(
                f'Building Model Complete...Total parameters: {self.model.total_parameters(var_list=var_to_save)}')
        self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
        logger.debug(f'Build Summary & Saver complete')
    
    def init_var(self):
        init = (var.initializer for var in tf.global_variables())
        self.sess.run(list(init))
        logger.info('Done initializing variables')
        
    def init_cv2_display():
        
        
    def close(self):
        self.sess.close()
        logger.info(f'Network shutdown!')

    def restore_model(self):
        if self.FLAGS.load_model_path is not None:
            logger.debug('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(check_point_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                logger.debug('Loading Model Succeeded...')
            except:
                logger.debug('Loading Model Failed')
        else:
            logger.debug('No Model to load')

    def save_model(self, name):
        self.saver.save(self.sess, f'./savedmodels/model-{name}.ckpt',
                        global_step=self.sess.run(self.model.global_step))
        
    def train_with_test(test=True):
    
    def test():
        pass
