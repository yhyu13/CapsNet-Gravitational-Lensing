import numpy as np
import tensorflow as tf
from tf_util import cost_tensor
from model.baseline import Baseline


class CNNBaseline(Baseline):

    def __init__(self, hps, images, labels):

        """
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, channels]
          labels: Batches of labels. [batch_size, num_classes]
        """
        super().__init__(hps, images, labels)

    def build_graph(self):
        """Build a whole graph for the model."""

        self._build_init()
        self.cost = self._build_model()

        grads_vars = self.optimizer.compute_gradients(self.cost)

        self._build_train_op(grads_vars)
        self.summaries = tf.summary.merge_all()

    """Overrride _build_model"""

    def _build_model(self):
        """Build the core model within the graph."""

        filters = [1, 128, 128, 64]
        assert isinstance(filters, list)
        strides = [5,1]
        assert isinstance(strides, list)
        cnn_kernel_size = [9,3]
        assert isinstance(cnn_kernel_size, list)
        padding = self.hps.padding

        with tf.variable_scope('init'):
            x = self.images
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[0],
                           filters[0], filters[1], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_1'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[0],
                           filters[1], filters[2], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after cnn_1 {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_2'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[1],
                           filters[2], filters[3], self._stride_arr(strides[1]), padding="SAME")
            x = self._relu(x)
            tf.logging.info('image after cnn_2 {}'.format(x.get_shape()))

        with tf.variable_scope('fc_1'):
            x = self._batch_norm('bn', x)
            x = self._fully_connected(x, 392)
            x = self._relu(x)
            tf.logging.info('image after fc_1 {}'.format(x.get_shape()))

        with tf.variable_scope('fc_2'):
            x = self._batch_norm('bn', x)
            x = tf.nn.dropout(x, keep_prob=0.5)
            x = self._fully_connected(x, self.hps.num_labels)
            #logits = tf.squeeze(x)
            #y_pred = tf.nn.softmax(logits)
            self.y_pred = tf.squeeze(x)
            tf.logging.info('image after fc_2 {}'.format(x.get_shape()))

        with tf.variable_scope('costs'):

            #ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true), name='prediction_loss')
            self.L, self.y_pred_flipped = cost_tensor(self.y_pred, self.labels)

            cost = self.L + self._decay()
            tf.summary.scalar('Prediction_loss', self.L)
            tf.summary.scalar('Total_loss', cost)

        return cost


    def _fully_connected(self, x, out_dim, name=''):
        """FullyConnected layer for final output."""
        x = tf.contrib.layers.flatten(x)
        w = tf.get_variable(
            name + 'DW', [x.get_shape()[1], out_dim],
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        b = tf.get_variable(name + 'biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.expand_dims(tf.nn.xw_plus_b(x, w, b), 2)


class CNNBaseline2(CNNBaseline):

    def __init__(self, hps, images, regress_labels, class_labels):
        super().__init__(hps, images, regress_labels)
        self.y_regress_labels = self.labels
        self.y_class_labels = class_labels

    """Overrride _build_model"""

    def _build_model(self):
        """Build the core model within the graph."""

        filters = [1, 128, 128, 64]
        assert isinstance(filters, list)
        strides = [5,1]
        assert isinstance(strides, list)
        cnn_kernel_size = [9,3]
        assert isinstance(cnn_kernel_size, list)
        padding = self.hps.padding

        with tf.variable_scope('init'):
            x = self.images
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[0],
                           filters[0], filters[1], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_1'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[0],
                           filters[1], filters[2], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after cnn_1 {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_2'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size[1],
                           filters[2], filters[3], self._stride_arr(strides[1]), padding="SAME")
            x = self._relu(x)
            tf.logging.info('image after cnn_2 {}'.format(x.get_shape()))

        with tf.variable_scope('fc_1'):
            x = self._batch_norm('bn', x)
            x = self._fully_connected(x, 392)
            x = self._relu(x)
            tf.logging.info('image after fc_1 {}'.format(x.get_shape()))

        with tf.variable_scope('fc_regress'):
            x = self._batch_norm('bn', x)
            y1 = tf.nn.dropout(x, keep_prob=0.5)
            y1 = self._fully_connected(y1, self.hps.num_labels)

            self.y_regress_pred = tf.squeeze(y1)

        with tf.variable_scope('fc_class'):
            y2 = tf.nn.dropout(x, keep_prob=0.5)
            y2 = self._fully_connected(y2, self.hps.task2_num_classes)

            self.y_class_pred = tf.sigmoid(tf.squeeze(y2))

        with tf.variable_scope('costs'):

            class_loss_mean, MSE, self.y_pred_flipped, class_loss_matrix = cost_tensor2(
                self.y_regress_pred, self.y_regress_labels, self.y_class_pred, self.y_class_labels, loss='logistic')

            self.L = class_loss_mean + MSE
            cost = self.L + self._decay()
            tf.summary.scalar('Regression_loss', MSE)
            tf.summary.scalar('Classification_loss', class_loss_mean)
            tf.summary.scalar('Total_prediction_loss', self.L)
            tf.summary.scalar('Total_loss', cost)

        with tf.variable_scope('acc'):
            # spread_loss entry equals to zero implies vector length is bigger/smaller than upper/lower margin
            # thus spread_loss entry equals to zero means correct prediction
            # for details, take a look at the implementation of cost_tensor2()
            correct_prediction = tf.equal(class_loss_mean, 0)
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='acc')
            tf.summary.scalar('accuracy', self.acc)

        return cost
