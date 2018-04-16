import argparse
import os
from collections import namedtuple

parser = argparse.ArgumentParser(description='Define parameters.')

"""Network hyperparameters"""
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--global_epoch', type=int, default=50)
parser.add_argument('--n_batch', type=int, default=50)
parser.add_argument('--n_img_row', type=int, default=192)
parser.add_argument('--n_img_col', type=int, default=192)
parser.add_argument('--n_img_channels', type=int, default=1)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_labels', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model', default='cap2', help='choose between cnn and cap')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--mode', dest='MODE', default='train', help='choose between train and test')

FLAGS = parser.parse_args()

"""CapsNet hyperparameters"""
HParams = namedtuple('HParams',
                     'batch_size, num_classes, num_labels '
                     'filters, strides, cnn_kernel_size, padding, '
                     'lambda_margin_loss,m_plus_margin_loss,m_minus_margin_loss, '
                     'num_routing, standard, label_masking, hist_summary, grad_defense, '
                     'weight_decay_rate, relu_leakiness, '
                     'fixed_lrn, min_lrn_rate, lrn_rate, decay_step, optimizer, temperature, global_norm, ')

HPS = HParams(batch_size=FLAGS.n_batch,
              num_classes=FLAGS.n_classes,
              num_labels=FLAGS.n_labels,
              filters=[1, 128, 8, 16],
              strides=[5, 5],
              cnn_kernel_size=9,
              padding="VALID",
              lambda_margin_loss=0.5,
              m_plus_margin_loss=0.9,
              m_minus_margin_loss=0.1,
              num_routing=3,
              standard=True,
              label_masking=True,
              hist_summary=False,
              grad_defense=False,
              weight_decay_rate=1e-4,
              relu_leakiness=0.0,
              fixed_lrn=True,
              min_lrn_rate=1e-5,
              lrn_rate=1e-3,
              decay_step=100,
              optimizer='adam',
              temperature=1.0,
              global_norm=100)

# https://github.com/yasharhezaveh/Ensai/blob/master/init.py
'''
   Defining global variables
'''

num_out = FLAGS.n_labels  # number ouf output parameters being predicted
numpix_side = FLAGS.n_img_row  # number of image pixels on the side

max_trainoise_rms = 0.1  # maximum rms of noise in training data
max_testnoise_rms = 0.1  # maximum rms of noise in test or validation data
max_noise_rms = max_testnoise_rms

max_psf_rms = 0.08 / 0.04  # maximum Gaussian PSF rms (in pixels)
max_cr_intensity = 0.5  # maximum scaling for cosmic ray and artefact maps

# if True, the noise rms will be chosen randomly for each sample with a max of max_noise_rms (above)
variable_noise_rms = True

num_training_samples = 50000
max_num_test_samples = 10000

cycle_batch_size = 50   # how many examples to read at a time (here it's equal to the batch size)
num_test_samples = max_num_test_samples#1000  # number of test samples

pix_res = 0.04  # pixel size in arcsec
L_side = pix_res * numpix_side

min_unmasked_flux = 0.75

# number of folders containing training or test data. If all 3 point to the same folder that's OK (only that folder will be used).
num_data_dirs = 3

# Get current working directory
cwd = os.getcwd()

arcs_data_path_1 = cwd + '/data/ARCS_2/ARCS_2/'
arcs_data_path_2 = cwd + '/data/ARCS_2/ARCS_2/'
arcs_data_path_3 = cwd + '/data/ARCS_2/ARCS_2/'
test_data_path_1 = cwd + '/data/SAURON_TEST/'
test_data_path_2 = cwd + '/data/SAURON_TEST/'
test_data_path_3 = cwd + '/data/SAURON_TEST/'

lens_data_path_1 = cwd + '/data/ARCS_2/ARCS_2/'
lens_data_path_2 = cwd + '/data/ARCS_2/ARCS_2/'
lens_data_path_3 = cwd + '/data/ARCS_2/ARCS_2/'
testlens_data_path_1 = cwd + '/data/SAURON_TEST/'
testlens_data_path_2 = cwd + '/data/SAURON_TEST/'
testlens_data_path_3 = cwd + '/data/SAURON_TEST/'

# folder containing cosmic rays
CRay_data_path = cwd + '/data/CosmicRays/'
# file containing guassian noise data
real_guassian_noise_path = cwd + '/data/PS_4_real.txt'
imag_guassian_noise_path = cwd + '/data/PS_4_imag.txt'
# xy range of center of the lens. The image is shifted in a central area with a side of max_xy_range (arcsec) during training or testing
max_xy_range = 0.5

# tensorflow logging folder
train_log_folder = cwd + '/train_log'
test_log_folder = cwd + '/test_log'

# When restoring a model, please change tensorflow checkpoint file under this path accordingly.
savedmodel_path = cwd + '/savedmodels/'
