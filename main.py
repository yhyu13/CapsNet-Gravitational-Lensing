from config import *
from dataset import *
from network import *


def train():
    net = Network(HPS, FLAGS)
    net.train(validation=True)


if __name__ == "__main__":

    func = {'train': lambda: train()}
    func[FLAGS.MODE]()
