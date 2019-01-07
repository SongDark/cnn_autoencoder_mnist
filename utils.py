import tensorflow as tf 
import numpy as np
from scipy.io import loadmat
import os

class BasicBlock(object):
    def __init__(self, hidden_units, name):
        self.name = name
        self.hidden_units = hidden_units
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

class BasicTrainFramework(object):
	def __init__(self, batch_size, version):
		self.batch_size = batch_size
		self.version = version

	def build_dirs(self):
		self.log_dir = os.path.join('logs', self.version) 
		self.model_dir = os.path.join('checkpoints', self.version)
		self.fig_dir = os.path.join('figs', self.version)
		for d in [self.log_dir, self.model_dir, self.fig_dir]:
			if (d is not None) and (not os.path.exists(d)):
				print "mkdir " + d
				os.makedirs(d)
	
	def build_sess(self):
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
	
	def build_network(self):
		self.D_logit_real = None 
		self.D_logit_fake = None

	def load_model(self, checkpoint_dir=None, ckpt_name=None):
		import re 
		print "load checkpoints ..."
		checkpoint_dir = checkpoint_dir or self.model_dir
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = ckpt_name or os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print "Success to read {}".format(ckpt_name)
			return True, counter
		else:
			print "Failed to find a checkpoint"
			return False, 0

def lrelu(x, leak=0.2, name='leaky_relu'):
	return tf.maximum(x, leak*x, name=name) 

def bn(x, is_training, name):
	return tf.contrib.layers.batch_norm(x, 
										decay=0.999, 
										updates_collections=None, 
										epsilon=0.001, 
										scale=True,
										fused=False,
										is_training=is_training,
										scope=name)

def spectral_norm(w, iteration=10, name="sn"):
	'''
	Ref: https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/65218e8cc6916d24b49504c337981548685e1be1/spectral_norm.py
	'''
	w_shape = w.shape.as_list() # [KH, KW, Cin, Cout] or [H, W]
	w = tf.reshape(w, [-1, w_shape[-1]]) # [KH*KW*Cin, Cout] or [H, W]

	u = tf.get_variable(name+"_u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
	s = tf.get_variable(name+"_sigma", [1, ], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u # [1, Cout] or [1, W]
	v_hat = None 

	for _ in range(iteration):
		v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w))) # [1, KH*KW*Cin] or [1, H]
		u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w)) # [1, Cout] or [1, W]
		
	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)) # [1,1]
	sigma = tf.reshape(sigma, (1,))

	with tf.control_dependencies([u.assign(u_hat), s.assign(sigma)]):
		# ops here run after u.assign(u_hat)
		w_norm = w / sigma 
		w_norm = tf.reshape(w_norm, w_shape)
	
	return w_norm

def linear(x, output_size, stddev=0.02, bias_start=0.0, name='linear'):
	shape = x.get_shape().as_list()
	with tf.variable_scope(name):
		W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(bias_start))

	return tf.matmul(x,W) + bias

def dense(x, output_size, stddev=0.02, bias_start=0.0, activation=None, sn=False, reuse=False, name='dense'):
	shape = x.get_shape().as_list()
	with tf.variable_scope(name, reuse=reuse):
		W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(bias_start))
		if sn:
			W = spectral_norm(W, name="sn")
	out = tf.matmul(x, W) + bias 
	if activation is not None:
		out = activation(out)
	
	return out

def conv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, padding="VALID", bias=True, name='conv2d'):
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, x.get_shape()[-1], channel], initializer=tf.truncated_normal_initializer(stddev=stddev))
		if sn:
			conv = tf.nn.conv2d(x, spectral_norm(w, name="sn"), strides=[1, d_h, d_w, 1], padding=padding)
		else:
			conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
		if bias:
			biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
			conv = tf.nn.bias_add(conv, biases)
	return conv

def deconv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, padding='VALID', name='deconv2d'):
	with tf.variable_scope(name):
		tmp = tf.get_variable('tmp', [k_h, k_w, channel, x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		
	def get_deconv_lens(H, k, d):
		if padding == "VALID":
			# return tf.multiply(H, d) + k - 1
			return H * d + k - 1
		elif padding == "SAME":
			# return tf.multiply(H, d)
			return H * d
	shape = tf.shape(x)
	H, W = shape[1], shape[2]
	N, _, _, C = x.get_shape().as_list()
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, channel, x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
		if sn:
			w = spectral_norm(w, name="sn")
	
	N, H, W, C = x.get_shape().as_list() # ???
	H1 = get_deconv_lens(H, k_h, d_h)
	W1 = get_deconv_lens(W, k_w, d_w)
	
	deconv = tf.nn.conv2d_transpose(x, w, output_shape=[N, H1, W1, channel], strides=[1, d_h, d_w, 1], padding=padding)
	deconv = tf.nn.bias_add(deconv, biases)
	return deconv

def conv_cond_concat(x, y):
    # x: [N, H, W, C]
    # y: [N, 1, 1, d]
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def one_hot_encode(ys, max_class):
    res = np.zeros((len(ys), max_class), dtype=np.float32)
    for i in range(len(ys)):
        res[i][ys[i]] = 1.0
    return res


def shuffle_in_unison_scary(*args, **kwargs):
	np.random.seed(kwargs['seed'])
	rng_state = np.random.get_state()
	for i in range(len(args)):
		np.random.shuffle(args[i])
		np.random.set_state(rng_state)

def mse(pred, data, num):
	loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / num 
	return loss_val

