"""CapsNet model.

Related papers:
https://arxiv.org/pdf/1710.09829.pdf
http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
"""
import numpy as np
import tensorflow as tf
from tf_util import cost_tensor
from model.baseline import Baseline


class CapsNet(Baseline):
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

    def _squash(self, x, axis=-1):
        """https://arxiv.org/pdf/1710.09829.pdf (eq.1)
           squash activation that normalizes vectors by their relative lengths
        """
        square_norm = tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)
        x = x * tf.sqrt(square_norm) / (1.0 + square_norm)
        return x

    def _capsule_layer(self, x, params_shape, num_routing, name=''):

        assert len(params_shape) == 4, "Given wrong parameter shape."
        input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule = params_shape

        # x = self._batch_norm(name + '/bn', x)

        # W.shape =  [None, input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule]
        W = tf.get_variable(
            name + '/capsule_layer_transformation_matrix', [
                1, input_num_capsule, input_dim_capsule, output_num_capsule, output_dim_capsule], tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

        # b.shape = [None, self.intput_num_capsule, 1, self.output_num_capsule, 1].
        b = tf.zeros([1, input_num_capsule, 1, output_num_capsule, 1])
        c = tf.nn.softmax(b, dim=3)

        # u.shape=[None, input_num_capsule, input_dim_capsule, 1, 1]
        u = tf.expand_dims(tf.expand_dims(x, -1), -1)
        u_ = tf.reduce_sum(u * W, axis=[2], keep_dims=True)
        u_stopped = tf.stop_gradient(u_)

        s = tf.reduce_sum(u_stopped * c, axis=[1], keep_dims=True)
        v = self._squash(s, axis=-1)
        tf.logging.info('Expanding inputs to be {}'.format(u.get_shape()))
        tf.logging.info(
            'Transforming and sum input capsule dimension (routing inputs){}'.format(u_.get_shape()))
        tf.logging.info('Outputs of each routing iteration {}'.format(v.get_shape()))

        assert num_routing > 1, 'The num_routing should be > 1.'

        for i in range(num_routing - 2):
            b += tf.reduce_sum(u_stopped * v, axis=-1, keep_dims=True)
            c = tf.nn.softmax(b, dim=3)
            s = tf.reduce_sum(u_stopped * c, axis=[1], keep_dims=True)
            v = self._squash(s, axis=-1)

        b += tf.reduce_sum(u_ * v, axis=-1, keep_dims=True)
        c = tf.nn.softmax(b, dim=3)
        s = tf.reduce_sum(u_ * c, axis=[1], keep_dims=True)
        v = self._squash(s, axis=-1)

        v_digit = tf.squeeze(v, axis=[1, 2])
        tf.logging.info('Image after this capsule layer {}'.format(v_digit.get_shape()))
        return v_digit

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
            x = self._conv('init_conv', x, 5, 1, 256, self._stride_arr(3), padding='VALID')
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))

        with tf.variable_scope('init2'):
            x = self._batch_norm('init_bn', x)
            x = self._conv('init_conv', x, 5, 256, 256, self._stride_arr(3), padding='VALID')
            x = self._relu(x)
            tf.logging.info('image after init {}'.format(x.get_shape()))

        with tf.variable_scope('primal_capsules'):
            x = self._batch_norm('primal_capsules_bn', x)
            x = self._conv('primal_capsules_conv', x, 5, 256, 256, self._stride_arr(3), padding='VALID')

            capsules_dims = 16
            num_capsules = np.prod(x.get_shape().as_list()[1:]) // capsules_dims

            x = tf.reshape(x, [-1, num_capsules, capsules_dims])
            x = self._squash(x, axis=-1)
            tf.logging.info('image after primal capsules {}'.format(x.get_shape()))

        with tf.variable_scope('digital_capsules_final'):
            """
                params_shape = [input_num_capsule,input_dim_capsule
                                output_num_capsule,output_dim_capsule]
            """
            params_shape = [num_capsules, capsules_dims, 64, 8]
            cigits = self._capsule_layer(x, params_shape=params_shape,
                                         num_routing=self.hps.num_routing, name='capsules_final_cigits')
            tf.logging.info('cigits shape {}'.format(cigits.get_shape()))

            self.y_pred = self._fully_connected(
                cigits, self.hps.num_labels, name='capsules_final_fc', dropout_prob=0.5)


        with tf.variable_scope('costs'):
            self.L, self.y_pred_flipped = cost_tensor(self.y_pred, self.labels)
            cost = self.L + self._decay()
            tf.summary.scalar('Prediction_loss', self.L)
            tf.summary.scalar('Total_loss', cost)

        return cost
