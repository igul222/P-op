import theano

from .base import Layer, Pop

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng = RandomStreams()


__all__ = [
    "DropoutLayer",
    "dropout",
    "GaussianNoiseLayer",
    "DropoutPop"
]
class DropoutPop(Pop):
    def __init__(self, p, **kwargs):
        """
        Here, p is dropout rate.
        """
        super(DropoutPop, self).__init__(1,1,**kwargs)
        self.p = p

    def symbolic_call(self, arg, **kwargs):
        if self.p == 0:
            print "DROPOUT IS NOTHING"
            return arg #this is just identity if p was 0.

        test = kwargs.get('test', False)
        if not test:
            print "COMPILING DROPOUT TRAIN-TIME"
            # scale arg; if we drop self.p of the arguments, everything should be divided by 1 - p, i.e usually the outputs are total, so we much find x s.t (1-p)inp/(x) = total, hence x=1-p.
            arg /= (1 - self.p)
            # make mask

            mask = _srng.binomial(arg.shape, p=1 -self.p, dtype=theano.config.floatX)
            return arg * mask
        else:
            print "COMPILING DROPOUT TEST-TIME"
            # here, don't do anything.
            return arg



class DropoutLayer(Layer):
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * _srng.binomial(input_shape, p=retain_prob,
                                          dtype=theano.config.floatX)

dropout = DropoutLayer  # shortcut


class GaussianNoiseLayer(Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            return input + _srng.normal(input.shape, avg=0.0, std=self.sigma)
