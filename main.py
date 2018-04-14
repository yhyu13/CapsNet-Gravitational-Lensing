from config import *
from dataset import *
from network import *


def train():
    # HPS, FLAGS defined in config.py
    net = Network(HPS, FLAGS)
    net.train(validation=True)
    
def test():
    # HPS, FLAGS defined in config.py
    net = Network(HPS, FLAGS)
    if FLAGS.restore:
        net.restore_model()
    net.test()

if __name__ == "__main__":

    func = {'train': lambda: train(),
            'test': lambda: test()}
    func[FLAGS.MODE]()
