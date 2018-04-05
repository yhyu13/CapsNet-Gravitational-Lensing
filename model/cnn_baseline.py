import numpy as np
import tensorflow as tf
from model.baseline import Baseline


class CNNBaseline(Baseline):
    """CapsNet model."""

    def __init__(self, hps, images, labels):
        """CapsNet constructor"""

        """
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, channels]
          labels: Batches of labels. [batch_size, num_classes]
        """
        super().__init__(hps, images, labels)

    def build_graph(self):
        """Build a whole graph for the model."""

        """
            1. create anxilliary parameters
            2. create CapsNet core layers
            3. create trainer
        """

        self._build_init()
        self.cost = self._build_model()

        grads_vars = self.optimizer.compute_gradients(self.cost)

        self._build_train_op(grads_vars)
        self.summaries = tf.summary.merge_all()

    """Overrride _build_model"""

    def _build_model(self):
        """Build the core model within the graph."""

        filters = [1, 256, 256, 128]
        assert isinstance(filters, list)
        strides = [1]
        assert isinstance(strides, list)
        cnn_kernel_size = 5
        assert isinstance(cnn_kernel_size, int)
        padding = self.hps.padding

        with tf.variable_scope('init'):
            x = self.images
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size,
                           filters[0], filters[1], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_1'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size,
                           filters[1], filters[2], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info('image after cnn_1 {}'.format(x.get_shape()))

        with tf.variable_scope('cnn_2'):
            x = self._batch_norm('bn', x)
            x = self._conv('conv', x, cnn_kernel_size,
                           filters[2], filters[3], self._stride_arr(strides[0]), padding=padding)
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
            x = self._fully_connected(x, self.hps.num_classes)
            logits = tf.squeeze(x)
            y_pred = tf.nn.softmax(logits)
            tf.logging.info('image after fc_2 {}'.format(x.get_shape()))

        with tf.variable_scope('costs'):
            """Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it."""
            y_true = self.labels

            ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_true), name='prediction_loss')

            cost = ce + self._decay()
            tf.summary.scalar('Total_loss', cost)
            tf.summary.scalar('Prediction_loss', ce)

        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.argmax(y_pred, 1), tf.argmax(self.labels, 1))
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')
            tf.summary.scalar('accuracy', self.acc)

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
