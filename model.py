import chainer.links as L
import chainer.functions as F
from chainer import dataset, Chain, training, optimizers, \
    iterators, reporter, cuda, serializers


class Net(Chain):

    def __init__(self, n_out):
        super(Net, self).__init__()

        with self.init_scope():
            self.l1 = L.ConvolutionND(
                ndim=1, in_channels=1,
                out_channels=1, ksize=3)
            self.l2 = L.ConvolutionND(
                ndim=1, in_channels=1,
                out_channels=1, ksize=3)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Loss_Link(Chain):
    def __init__(self, model):
        super(Loss_Link, self).__init__()
        self.y = None
        with self.init_scope():
            self.model = model

    def __call__(self, x, t):
        self.y = self.model(x)
        self.mean_loss = F.mean_squared_error(self.y, t)
        reporter.report({'mean_loss': self.mean_loss}, self)
        self.worst_loss = F.max(F.squared_error(self.y, t))
        reporter.report({'worst_loss': self.worst_loss}, self)
        return self.mean_loss
