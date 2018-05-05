import tensorflow as tf
from config import numpix_side, num_out, HPS


def cost_tensor(y_conv, y_label):
    FLIPXY = tf.constant([[1., 0., 0., 0., 0.], [0., -1., 0., 0., 0.],
                          [0., 0., -1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    scale_par_cost = tf.constant([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [
                                 0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    y_conv_flipped = tf.matmul(y_conv, FLIPXY)
    scaled_delta_1 = tf.matmul(tf.pow(y_conv - y_label, 2), scale_par_cost)
    scaled_delta_2 = tf.matmul(tf.pow(y_conv_flipped - y_label, 2), scale_par_cost)
    MSE = tf.reduce_mean(tf.minimum(tf.reduce_mean(scaled_delta_1, axis=1),
                                    tf.reduce_mean(scaled_delta_2, axis=1)), axis=0)
    return MSE, y_conv_flipped


def cost_tensor2(y_regress_conv, y_regress_label, y_class_conv, y_class_label):
    FLIPXY = tf.constant([[1., 0., 0., 0., 0.], [0., -1., 0., 0., 0.],
                          [0., 0., -1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    scale_par_cost = tf.constant([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [
                                 0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    y_conv_flipped = tf.matmul(y_conv, FLIPXY)
    scaled_delta_1 = tf.matmul(tf.pow(y_conv - y_label, 2), scale_par_cost)
    scaled_delta_2 = tf.matmul(tf.pow(y_conv_flipped - y_label, 2), scale_par_cost)
    MSE = tf.minimum(tf.reduce_mean(scaled_delta_1, axis=1), tf.reduce_mean(scaled_delta_2, axis=1))
    regress_loss_mask = tf.reduce_sum(tf.multiply(tf.constant([[0, 1]]), y_class_label), axis=1)
    MSE = tf.reduce_mean(MSE * regress_loss_mask, axis=0)

    # classification loss
    lbd = HPS.lambda_margin_loss  # 0.5
    m_plus = HPS.m_plus_margin_loss  # 0.9
    m_minus = HPS.m_minus_margin_loss  # 0.1
    Spread_loss = y_class_label * tf.square(tf.maximum(0., m_plus - y_class_conv)) + \
        lbd * (1 - y_class_label) * tf.square(tf.maximum(0., y_class_conv - m_minus))

    Spread_loss = tf.reduce_mean(tf.reduce_sum(Spread_loss, 1))

    return Spread_loss + MSE, y_conv_flipped, spread_loss


def init_xy_placeholder():
    # https://github.com/yhyu13/Ensai/blob/refactory/model/ensai_model.py
    # placeholder for input image
    x = tf.placeholder(tf.float32, shape=[None, numpix_side * numpix_side])
    # placeholder for output parameters during training
    y_label = tf.placeholder(tf.float32, shape=[None, num_out])
    x_image0 = tf.reshape(x, [-1, numpix_side, numpix_side, 1])
    # removing image intensity bias: filter image with a 4X4 filter and remove from image
    MASK = tf.abs(tf.sign(x_image0))
    XX = x_image0 + ((1 - MASK) * 1000.0)
    bias_measure_filt = tf.constant((1.0 / 16.0), shape=[4, 4, 1, 1])
    bias_measure = tf.nn.conv2d(XX, bias_measure_filt, strides=[1, 1, 1, 1], padding='VALID')
    im_bias = tf.reshape(tf.reduce_min(bias_measure, axis=[1, 2, 3]), [-1, 1, 1, 1])
    x_image = x_image0 - (im_bias * MASK)
    return x, y_label, x_image
