import tensorflow as tf


def cost_tensor(y_conv, y_label):
	FLIPXY = tf.constant([ [1., 0., 0., 0., 0.],[0. , -1. , 0., 0., 0.],[0., 0. , -1., 0., 0.],[0., 0. , 0., 1., 0.],[0., 0. , 0., 0., 1.]] )
	scale_par_cost =  tf.constant([[1., 0., 0., 0., 0.],[0. , 1. , 0., 0., 0.],[0., 0. , 1., 0., 0.],[0., 0. , 0., 1., 0.],[0., 0. , 0., 0., 1.]])
	y_conv_flipped = tf.matmul(y_conv, FLIPXY)
	scaled_delta_1 = tf.matmul(tf.pow(y_conv - y_label,2) , scale_par_cost)
	scaled_delta_2 = tf.matmul(tf.pow(y_conv_flipped - y_label,2) , scale_par_cost)
	MeanSquareCost = tf.reduce_mean( tf.minimum(tf.reduce_mean( scaled_delta_1 ,axis=1) , tf.reduce_mean(  scaled_delta_2 ,axis=1)) , axis=0)
	return MeanSquareCost, y_conv_flipped



