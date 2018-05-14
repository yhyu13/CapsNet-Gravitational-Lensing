import argparse
import argh
import sys
import os
import tensorflow as tf
import numpy as np

import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

from numpy.random import RandomState

prng = RandomState(1234567890)

from matplotlib import pyplot as plt
import cv2

from dataset import *


def plot_imgs(inputs, num, label):
    """Plot smallNORB images helper"""
    # fig = plt.figure()
    # plt.title('Show images')
    # r = np.floor(np.sqrt(len(inputs))).astype(int)
    # for i in range(r**2):
    #     size = inputs[i].shape[1]
    #     sample = inputs[i].flatten().reshape(size, size)
    #     a = fig.add_subplot(r, r, i + 1)
    #     a.imshow(sample, cmap='gray')
    # plt.show()
    inputs = (inputs).astype(np.float32)
    for i in range(len(inputs)):
        size = inputs[i].shape[1]
        cv2.imwrite('./imgs'+'%d' % num+'_%d' % i+label+'.jpg', inputs[i].flatten().reshape(size, size))
    return


def write_data_to_tfrecord(kind:str, name='',batch_size=50, num_chunks = 10):

    from time import time
    start = time()

    """Read data"""
    if kind == "train":
        total_num_images = num_training_samples // num_chunks
    elif kind == "test":
        total_num_images = max_num_test_samples // num_chunks
    else:
        logger.warning('Please choose either training or testing data to preprocess.')

    """Write to tfrecord"""
    for k in range(num_chunks):
        writer = tf.python_io.TFRecordWriter("./data/" + kind + "_%s_%d.tfrecords" % (name,k))
        num_batch = total_num_images // batch_size

        for j in range(k*num_batch, (k+1)*num_batch):
            images, labels, _ = read_data_batch(j, batch_size, kind)

            for i in range(batch_size):
                if i % batch_size == 0:
                    logger.info('Write ' + kind + ' images %d' % (j * batch_size))
                img = images[i]
                lab = labels[i]
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=lab)),
                    'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()
        logger.info('Done writing %dth ' + kind + 'record . Elapsed time: %f' % (time() - start))
        start = time()


def tfrecord():
    """Wrapper"""
    write_data_to_tfrecord(kind='train', name='task1')
    write_data_to_tfrecord(kind='test', name='task1')
    logger.info('Writing train & test to TFRecord done.')


def read_tfrecord(filenames, epochs: int):

    assert isinstance(filenames, list)

    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature(shape=[num_out],dtype=tf.float32),
                                           'img_raw': tf.FixedLenFeature(shape=[numpix_side*numpix_side],dtype=tf.float32),
                                       })
    img = tf.cast(features['img_raw'],tf.float32)
    label = tf.cast(features['label'],tf.float32)
    logger.info('Raw->img shape: {}, label shape: {}'.format(img.get_shape(), label.get_shape()))
    return img, label


def test(is_train=True):
    """Instruction on how to read data from tfrecord"""

    # 1. use regular expression to find all files we want
    import re
    if is_train:
        CHUNK_RE = re.compile(r"train.+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test.+\.tfrecords")

    processed_dir = './data'
    # 2. parse them into a list of file name
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    # 3. pass argument into read method
    logger.info('Read from {}'.format(chunk_files))
    image, label = read_tfrecord(chunk_files, 2)

    batch_size = 50
    x, y = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)
    logger.info('x shape: {}, y shape: {}'.format(x.get_shape(), y.get_shape()))

    # 初始化所有的op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        # 启动队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(2):
            val, l = sess.run([x, y])
            # l = to_categorical(l, 12)
            print(val, l)
        coord.join()

    logger.info('Test read tf record Succeed')

if __name__ == "__main__":
    test(is_train=True)
    test(is_train=False)
