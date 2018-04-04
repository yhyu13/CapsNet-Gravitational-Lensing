from config import *
from dataset import read_data_batch, get_rotation_corrected
from model.capsnet_model import CapsNet
from model.cnn_baseline import CNNBaseline

from tf_util import init_xy_placeholder

import os
import sys
import logging
import daiquiri
import tensorflow as tf
import numpy as np

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

class Network:

    def __init__(self, hps, FLAGS):
        self.hps = hps
        self.FLAGS = FLAGS
        
        self.num_batch = self.FLAGS.n_batch
        tf.reset_default_graph()
        tf.set_random_seed(1234)
        g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.33
        self.sess = tf.Session(config=config, graph=g)
        
        with g.as_default():
            self.init_preprocess()
            self.init_model()
            self.init_summary()
            self.init_var()
    
    def init_preprocess(self):
        self.x, self.y_label, self.x_image = init_xy_placeholder()
        
    def init_model(self):
        models = {'cnn': lambda: CNNBaseline(self.hps, self.x_image, self.y_label),
                  'cap': lambda: CapsNet(self.hps, self.x_image, self.y_label)}
        self.model = models[self.FLAGS.model]()
        logger.info("Building Model...")
        self.model.build_graph()
        
    def init_summary(self):
        self.summary = self.model.summaries
        self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./test_log")
        
        var_to_save = tf.trainable_variables() + [var for var in tf.global_variables()
                                                      if ('bn' in var.name) and ('Adam' not in var.name) and ('Momentum' not in var.name) or ('global_step' in var.name)]
        logger.info(
                'Building Model Complete...Total parameters: {}'.format(self.model.total_parameters(var_list=var_to_save)))
        self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
        logger.info(f'Build Summary & Saver complete')
    
    def init_var(self):
        init = (var.initializer for var in tf.global_variables())
        self.sess.run(list(init))
        logger.info('Done initializing variables')
        
    def init_cv2_display(self):
        # import cv2
        pass
        
    def close(self):
        self.sess.close()
        logger.info(f'Network shutdown!')
        sys.exit()

    def restore_model(self):
        if self.FLAGS.load_model_path is not None:
            logger.info('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(check_point_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                logger.info('Loading Model Succeeded...')
            except:
                logger.info('Loading Model Failed')
        else:
            logger.info('No Model to load')

    def save_model(self, name):
        self.saver.save(self.sess, './savedmodels/model-{}.ckpt'.format(name),
                        global_step=self.sess.run(self.model.global_step))
        
    def train(self, porportion=1.0, validation=False):
        logger.info('Train model...')
        num_iter = int(num_training_samples * porportion // self.num_batch)
        logger.info('1 Epoch training steps will be: {}'.format(num_iter))
        
        save_per_iter = num_iter / 10
        
        for i in range(num_iter):
        
            if validation and (i % 20 == 1):
                self.test(porportion=10, random_sample=True)
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
                    logger.info('Train step {} | Loss: {:.3f} | Global step: {}'.format(i,l,global_step))
                #if (i+1) % save_per_iter == 0:
                    #self.save_model(name="{}".format(global_step))

    def test(self, porportion=1.0, random_sample = False):
        logger.info('Test model...')
        RMS_moving = None
        Loss_moving = None
        if porportion > 1:
            """Validation"""
            num_iter = int(porportion)
            RMS_moving = 0.0
            Loss_moving = 0.0
        else:
            num_iter = int(num_training_samples * porportion // self.num_batch)
        
        logger.info('Testing steps will be: {}'.format(num_iter))
        
        for i in range(num_iter):
        
            X,Y,_ = read_data_batch(indx=np.random.randint(0, high=num_test_samples // self.num_batch) if max_num_test_samples else i, batch_size=self.num_batch, train_or_test="test")
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
                self.test_writer.add_summary(summary, global_step)
                ROT_COR_PARS = get_rotation_corrected(y_pred,y_pred_flipped,Y)
                RMS = np.std(ROT_COR_PARS-Y,axis=0)
                if RMS_moving is not None and Loss_moving is not None:
                    """Validation"""
                    RMS_moving += RMS
                    Loss_moving += l
                    if i == num_iter-1:
                        logger.info('Moving LOSSS: {:.3f} | Moving RMS: {}'.format(Loss_moving / num_iter, np.array_str(RMS_moving / num_iter, precision=3)))
                else:
                    logger.info('Test step {} | LOSS: {:.3f} | RMS: {}'.format(i,l,np.array_str(RMS,precision=3)))
