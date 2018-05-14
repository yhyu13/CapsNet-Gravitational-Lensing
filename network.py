from config import *
from dataset import read_data_batch, get_rotation_corrected
from model.capsnet_model import CapsNet
from model.cnn_baseline import CNNBaseline

from tf_util import init_x_image
from tf_record import read_tfrecord

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
        # 1. use regular expression to find all files we want
        import re
        if self.FLAGS.mode == 'train':
            CHUNK_RE = re.compile(r"train.+\.tfrecords")
        elif self.FLAGS.mode == 'test':
            CHUNK_RE = re.compile(r"test.+\.tfrecords")
        else:
            self.close()

        processed_dir = './data'
        # 2. parse them into a list of file name
        chunk_files = [os.path.join(processed_dir, fname)
                       for fname in os.listdir(processed_dir)
                       if CHUNK_RE.match(fname)]
        # 3. pass argument into read method
        logger.info('Read from {}'.format(chunk_files))
        image, label = read_tfrecord(chunk_files, self.FLAGS.global_epoch)

        batch_size = self.num_batch
        self.x, self.y_label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=batch_size * 64,
                                      min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)
        self.x_image = init_x_image(self.x)

    def init_model(self):
        models = {'cnn': lambda: CNNBaseline(self.hps, self.x_image, self.y_label),
                  'cap': lambda: CapsNet(self.hps, self.x_image, self.y_label)}
        self.model = models[self.FLAGS.model]()
        logger.info("Building Model...")
        self.model.build_graph()

    def init_summary(self):
        self.summary = self.model.summaries
        self.train_writer = tf.summary.FileWriter(train_log_folder)
        self.test_writer = tf.summary.FileWriter(test_log_folder)

        var_to_save = [var for var in tf.global_variables() if ('Adam' not in var.name) and ('Momentum' not in var.name)]
        logger.info(
            'Building Model Complete...Total parameters: {}'.format(self.model.total_parameters(var_list=var_to_save)))
        self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
        logger.info('Build Summary & Saver complete')

    def init_var(self):
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        logger.info('Done initializing variables')

    def init_cv2_display(self):
        # import cv2
        pass

    def close(self):
        self.sess.close()
        logger.info('Network shutdown!')
        sys.exit()

    def restore_model(self):
        if savedmodel_path is not None:
            logger.info('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(savedmodel_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                logger.info('Loading Model Succeeded...')
            except:
                logger.info('Loading Model Failed')
        else:
            logger.info('No Model to load')

    def save_model(self, name=""):
        logger.info('Saving model...')
        # savedmodel_path is defined in config.py
        self.saver.save(self.sess, savedmodel_path + 'model-{}.ckpt'.format(name),
                        global_step=self.sess.run(self.model.global_step))

    def train(self, porportion=1.0, validation=False):
        logger.info('Train model...')
        num_iter = int(self.FLAGS.global_epoch * num_training_samples * porportion // self.num_batch)
        logger.info('1 Epoch training steps will be: {}'.format(num_iter // self.FLAGS.global_epoch))

        # save per 10% of training
        save_per_iter = num_iter // 10

        for i in range(num_iter):

            feed_dict = {self.model.is_training: True}    # https://github.com/yhyu13/Ensai/blob/refactory/model/ensai_model.py
            try:
                _, summary, l, y_pred, y_pred_flipped, Y = \
                    self.sess.run([self.model.train_op, self.summary,
                                   self.model.L, self.model.y_pred, self.model.y_pred_flipped, self.y_label], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()

            except tf.errors.InvalidArgumentError:
                continue
            else:
                global_step = self.sess.run(self.model.global_step)
                self.sess.run(self.model.increase_global_step)
                if i % 5 == 0:
                    self.train_writer.add_summary(summary, global_step)
                    ROT_COR_PARS = get_rotation_corrected(y_pred, y_pred_flipped, Y)
                    RMS = np.std(ROT_COR_PARS - Y, axis=0)
                    logger.info('Train step {} | MSE: {:.3f} | RMS: {} | Global step: {}'.format(
                        i, l,  np.array_str(RMS, precision=3), global_step))
                if (i + 1) % save_per_iter == 0:
                    self.save_model(name=self.FLAGS.model)

    def test(self, porportion=1.0, random_sample=False):
        logger.info('Test model...')

        # init log file that contains RMS records
        log_file = open("log_file.txt", "w")
        log_file.close()

        RMS_moving = [0.0]*self.FLAGS.n_labels
        Loss_moving = 0.0

        num_iter = int(num_test_samples * porportion // self.num_batch)

        logger.info('Testing steps will be: {}'.format(num_iter))

        for i in range(num_iter):
            feed_dict = {self.model.is_training: False}
            try:
                y_pred, y_pred_flipped, Y, summary, l = \
                    self.sess.run([self.model.y_pred, self.model.y_pred_flipped, self.y_label,
                                   self.summary, self.model.L], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()

            except tf.errors.InvalidArgumentError:
                continue
            else:
                ROT_COR_PARS = get_rotation_corrected(y_pred, y_pred_flipped, Y)
                RMS = np.std(ROT_COR_PARS - Y, axis=0)
                RMS_moving += RMS
                Loss_moving += l

                global_step = self.sess.run(self.model.global_step)
                self.test_writer.add_summary(summary, global_step)
                logger.info('Moving MSE: {:.3f} | Moving RMS: {}'.format(Loss_moving / (i + 1), np.array_str(RMS_moving / (i + 1), precision=3)))
                #log_file = open("log_file.txt","a")
                #log_file.write('{} '.format(i) + ' '.join(map(str,[round(i,5) for i in RMS])) + ' {.5f}\n'.format(l) )
                # log_file.close()
