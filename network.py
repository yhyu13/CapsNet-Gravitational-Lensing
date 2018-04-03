from config import *
from model import *

from tf_util import init_xy_placeholder

import os
import sys
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

class Network:

    def __init__(self, hps, FLAGS):
        self.hps = hps
        self.FLAGS = FLAGS
        
        tf.reset_default_graph()
        tf.set_random_seed(1234)
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
                'Building Model Complete...Total parameters: {}'.format(self.model.total_parameters(var_list=var_to_save)))
        self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
        logger.debug(f'Build Summary & Saver complete')
    
    def init_var(self):
        init = (var.initializer for var in tf.global_variables())
        self.sess.run(list(init))
        logger.info('Done initializing variables')
        
    def init_cv2_display():
        # import cv2
        pass
        
    def close(self):
        self.sess.close()
        logger.info(f'Network shutdown!')
        sys.exit()

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
        self.saver.save(self.sess, './savedmodels/model-{}.ckpt'.format(name),
                        global_step=self.sess.run(self.model.global_step))
        
    def train(self, porportion=1.0, validation=False):
        logger.info('Train model...')
        num_iter = int(num_training_samples * porportion // self.num_batch) + 1
        logger.info('1 Epoch training steps will be: {}'.format(num_iter))
        
        for i in range(num_iter):
        
            if validation and (i % 20 == 0):
                self.test(porportion=0, random_sample=True)
                continue
            x,y,_ = read_data_batch(indx=i, batch_size=self.num_batch, train_or_test="train")
            feed_dict = {self.x: x,
                         self.y_label: y,
                         self.model.is_training: True}
            try:
                _, summary, l = \
                    self.sess.run([self.model.train_op, self.summary, self.model.cost], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()
                
            except tf.errors.InvalidArgumentError:
                continue
            else:
                global_step = self.sess.run(self.model.global_step)
                self.train_writer.add_summary(summary, global_step)
                self.sess.run(self.model.increase_global_step)
                if i % 2 == 0:
                    logger.debug('Train step {} | Loss: {:.3f} | Global step: {}'.format(i,l,global_step))

    def test(porportion=1.0, random_sample = False):
        logger.info('Test model...')
        num_iter = int(num_training_samples * porportion // self.num_batch) + 1
        logger.info('Testing steps will be: {}'.format(num_iter))
        
        for i in range(num_iter):
        
            X,Y,_ = read_data_batch(indx=np.random.randint(0, high=num_test_samples) if max_num_test_samples else i, batch_size=self.num_batch, train_or_test="test")
            feed_dict = {self.x: X,
                         self.y_label: Y,
                         self.model.is_training: False}
            try:
                y_pred, y_pred_flipped, summary, l = \
                    self.sess.run([self.model.y_pred, self.model.y_pred_flipped, self.summary, self.model.cost], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()
                
            except tf.errors.InvalidArgumentError:
                continue
            else:
                global_step = self.sess.run(self.model.global_step)
                self.test_write.add_summary(summary, global_step)
                ROT_COR_PARS = get_rotation_corrected(y_pred,y_pred_flipped,Y)
                RMS = np.std(ROT_COR_PARS-Y,axis=0)
                logger.debug('Test step {} | LOSS: {:3.f} | RMS: {}'.format(i,l,np.array_str(RMS,precision=2)))
