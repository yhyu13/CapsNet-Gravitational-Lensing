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

def train2():
    pass

def test2():
    pass

if __name__ == "__main__":

    if FLAGS.task2:
        func = {'train': lambda: train(),
                'test': lambda: test()}
    else:
        func = {'train': lambda: train2(),
                'test': lambda: test2()}
    func[FLAGS.MODE]()
