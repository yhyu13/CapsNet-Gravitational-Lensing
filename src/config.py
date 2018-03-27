import argparse
from collections import namedtuple


parser = argparse.ArgumentParser(description='Define parameters.')

"""Network hyperparameters"""
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--global_epoch', type=int, default=50)
parser.add_argument('--n_batch', type=int, default=64)
parser.add_argument('--n_img_row', type=int, default=28)
parser.add_argument('--n_img_col', type=int, default=28)
parser.add_argument('--n_img_channels', type=int, default=1)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_labels', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--fixed_lr', action='store_true')
parser.add_argument('--model', default='cap', help='choose between cnn and cap')
parser.add_argument('--dataset', dest='processed_dir', default='./MNIST_data')
parser.add_argument('--load_model_path', dest='load_model_path', default=None)  # './savedmodels'
parser.add_argument('--mode', dest='MODE', default='train', help='choose between train and test')

FLAGS = parser.parse_args()

"""CapsNet hyperparameters"""
HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, decay_step, '
                     'filters, strides, cnn_kernel_size, padding, '
                     'lambda_margin_loss,m_plus_margin_loss,m_minus_margin_loss, '
                     'num_routing, standard, label_masking, '
                     'weight_decay_rate, relu_leakiness, optimizer, temperature, global_norm, ')

HPS = HParams(batch_size=FLAGS.n_batch,
              num_classes=FLAGS.n_classes,
              num_labels=FLAGS.n_labels,
              fixed_lrn=FLAGS.fixed_lrn,
              min_lrn_rate=1e-5,
              lrn_rate=1e-3,
              decay_step=100,
              filters=[1, 256, 32, 16],
              strides=[1, 2],
              cnn_kernel_size=9,
              padding="VALID",
              lambda_margin_loss=0.5,
              m_plus_margin_loss=0.9,
              m_minus_margin_loss=0.1,
              num_routing=3,
              standard=True,
              label_masking=True,
              weight_decay_rate=1e-4,
              relu_leakiness=0.0,
              optimizer='adam',
              temperature=1.0,
              global_norm=100)
