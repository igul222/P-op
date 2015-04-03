from .base import *
import theano
import theano.tensor as T


class SumPop(Pop):
	def __init__(self, **kwargs):
		super(SumPop,self).__init__(num_input=2, num_output=1, **kwargs)

	def symbolic_call(self, one, two):
		return one + two


class DoublePop(Pop):
	def __init__(self, **kwargs):
		super(DoublePop,self).__init__(1,1,**kwargs)

	def symbolic_call(self, inp):
		return inp*2


