"""CapsNet model.

Related papers:
https://arxiv.org/pdf/1710.09829.pdf
http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
"""
import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages


class Baseline(object):
    """CapsNet model."""

    def __init__(self, hps, images, labels):
        """CapsNet constructor"""

        """
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, channels]
          labels: Batches of labels. [batch_size, num_classes]
        """
        self.hps = hps
        self.images = images
        self.labels = labels

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""

        """
        self._build_init()
        self.cost = self._build_model()

        grads_vars = self.optimizer.compute_gradients(self.cost)

        self._build_train_op(grads_vars)
        self.summaries = tf.summary.merge_all()
        """
        pass

    def _build_init(self):
        with tf.device('/cpu:0'):

            self.is_training = tf.placeholder(tf.bool)

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.increase_global_step = self.global_step.assign_add(1)

            if not self.hps.fixed_lrn:
                self.lrn_rate = tf.maximum(tf.train.exponential_decay(
                    self.hps.lrn_rate, self.global_step, 1e3, 0.66), self.hps.min_lrn_rate)
            else:
                self.lrn_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, trainable=False)
            tf.summary.scalar('learning_rate', self.lrn_rate)

            if self.hps.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lrn_rate)

    def _build_model(self):
        """Build the core model within the graph.

            Must overrride.
        """
        """
        filters = self.hps.filters
        assert isinstance(filters, list)
        strides = self.hps.strides
        assert isinstance(strides, list)
        cnn_kernel_size = self.hps.cnn_kernel_size
        assert isinstance(cnn_kernel_size, int)
        padding = self.hps.padding

        with tf.variable_scope('init'):
            x = self.images
            x = self._batch_norm('init_bn', x)
            x = self._conv('init_conv', x, cnn_kernel_size,
                           filters[0], filters[1], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info(f'image after init {x.get_shape()}')

        with tf.variable_scope('final_layer'):
            logits = self._fully_connected(x, self.hps.num_classes)

        with tf.variable_scope('costs'):
            y_true = self.labels
            ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
            cost = ce + self._decay()
            tf.summary.scalar(f'Total_loss', cost)
            tf.summary.scalar(f'Prediction_loss', ce)

        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.argmax(y_pred, 1), tf.argmax(self.labels, 1))
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')
            tf.summary.scalar(f'accuracy', self.acc)

        return cost
        """
        pass

    def _build_train_op(self, grads_vars):
        """Build training specific ops for the graph."""
        # Add histograms for trainable variables.
        # Add histograms for gradients.
        if self.hps.grad_summary:
            for grad, var in grads_vars:
                if grad is not None:
                    tf.summary.histogram(var.op.name, var)
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        if self.hps.grad_defense:
            # defensive step 2 to clip norm
            clipped_grads, self.norm = tf.clip_by_global_norm(
                [g for g, _ in grads_vars], self.hps.global_norm)

            # defensive step 3 check NaN
            # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
            grad_check = [tf.check_numerics(g, message='NaN Found!') for g in clipped_grads]
            with tf.control_dependencies(grad_check):
                apply_op = self.optimizer.apply_gradients(
                    zip(clipped_grads, [v for _, v in grads_vars]),
                    global_step=self.global_step, name='train_step')

                train_ops = [apply_op] + self._extra_train_ops
                # Group all updates to into a single train op.
                self.train_op = tf.group(*train_ops)
        else:
            apply_op = self.optimizer.apply_gradients(grads_vars,
                    global_step=self.global_step, name='train_step')

            train_ops = [apply_op] + self._extra_train_ops
            # Group all updates to into a single train op.
            self.train_op = tf.group(*train_ops)

    """Anxilliary methods"""

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.99))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.99))
            tf.summary.histogram(moving_mean.op.name, moving_mean)
            tf.summary.histogram(moving_variance.op.name, moving_variance)

            def train():
                # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

            def test():
                return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001)
            y = tf.cond(tf.equal(self.is_training, tf.constant(True)), train, test)
            y.set_shape(x.get_shape())
            return y

    # override _conv to use He initialization with truncated normal to prevent dead neural
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, padding='VALID'):
        """Convolution."""
        with tf.variable_scope(name):
            # n = filter_size * filter_size * out_filters
            n = in_filters + out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.truncated_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _relu(self, x, leakiness=0.0, elu=False):
        """Relu, with optional leaky support."""
        if leakiness > 0.0:
            return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        elif elu:
            return tf.nn.elu(x)
        else:
            return tf.nn.relu(x)

    def _fully_connected(self, x, out_dim, name=''):
        """FullyConnected layer for final output."""
        x = tf.contrib.layers.flatten(x)
        w = tf.get_variable(
            name + 'DW', [x.get_shape()[1], out_dim],
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        b = tf.get_variable(name + 'biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def total_parameters(self, var_list=None):
        if var_list is None:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() + [var for var in tf.global_variables() if 'bn' in var.name]]).astype(np.int32)
        else:
            return np.sum([np.prod(v.get_shape().as_list()) for v in var_list]).astype(np.int32)
