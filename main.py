from config import *
from dataset import *
from network import *


def train():
    # HPS, FLAGS defined in config.py
    net = Network(HPS, FLAGS)
    if FLAGS.restore:
        net.restore_model()
    net.train(porportion=0.1)

def test():
    # HPS, FLAGS defined in config.py
    net = Network(HPS, FLAGS)
    if FLAGS.restore:
        net.restore_model()
    net.test(porportion=0.1)

if __name__ == "__main__":

    func = {'train': lambda: train(),
            'test': lambda: test()}
    func[FLAGS.mode]()
