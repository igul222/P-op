import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities
from .. import utils

from .base import Pop
from .noise import DropoutPop


__all__ = [
    "FFNetPop",
    "TensorPop"
]

class TensorPop(Pop):
	"""
	This is a Pop that performs a tensor-product as in the Recursive Neural Tensor Network paper:

	v_out = a * V * a,  + W*a + b, where V is output_size x input_size x input_size. The dimension d of v_out is 
	the d'th slice of T (V[d]) as follows: a^T * (V[d]) * a
	"""
	def __init__(self, input_size, output_size, V=init.Uniform(), W=init.Uniform(), b=init.Uniform(), nonlinearity=nonlinearities.identity, **kwargs):
		"""
		Initializes the Pop

		Parameters
		----------
		input_size: dimensionality of the vectors that are input to the tensor
		output_size: desired dimensionalty of the output vector
		V: initializations for the tensor V
		b: initialization of the bias for the vector output
		"""
		super(TensorPop,self).__init__(1,1,**kwargs)
		self.V = self.create_param(V, (output_size,input_size,input_size), name="V")
		self.W = self.create_param(W, (input_size, output_size), name="W")
		self.b = self.create_param(b, (output_size,), name="b")

		self.nonlinearity = nonlinearity

	def get_params(self):
		return [self.V] + self.get_bias_params()

	def get_bias_params(self):
		return [self.b]

	def symbolic_call(self, vec, **kwargs):
		"""
		Performs the computation vec * V * vec
		"""
		outer = T.outer(vec,vec)
		tens_out = T.tensordot(self.V, outer, axes=[[1,2],[0,1]])
		matr_out = T.dot(vec, self.W)
		return self.nonlinearity(tens_out + matr_out + self.b.dimshuffle('x',0))



class FFNetPop(Pop):
	"""
	This is a Pop that makes an arbitrarily-sized neural net given the init statements. It only supports feedforward neural networks.
	"""
	def __init__(self, input_size, output_size, intermediary_sizes, initializations=(init.Uniform(), init.Constant(0.)), dropout=0., Wio=init.Uniform(), bo = init.Constant(0.), intermediary_nonlinearities=nonlinearities.rectify, output_nonlinearity=nonlinearities.identity, **kwargs):
		"""
		Initializes the Pop

		Parameters
		----------
		input_size: dimensionality of the bottom-level input
		intermediary_sizes: list of intermediary dimensionalities for the hidden states. This MUST BE A LIST in order to get #layers.
		initialization_list: a list of (W, b) pairs that are valid initializations (Lasagne-like) for the feed forward network, or just one init to use everywhere 
		nonlinearity_list: a list of nonlinearities to use for activations for each layer, or just one nonlinearity to use everywhere
		"""
		super(FFNetPop, self).__init__(1,1,**kwargs)
		# assert that all the lists are the same size or not a list
		# TODO: this is sorta hacky, try to make look nicer
		listsize = len(intermediary_sizes)
		for x in [initializations, intermediary_nonlinearities]:
			if isinstance(x, list):
				if len(x) != listsize:
					raise ValueError("%s was not the same size as intermediary_sizes. Expected: %s. got: %s" % (x, listsize, len(x)))
		
		#now create params
		# this either returns the thing itself if it's a list of size listsize, or makes it a list of size listsize, or raises an error.
		initializations = utils.stretch_to_list(initializations, listsize)
		self.intermediary_nonlinearities = utils.stretch_to_list(intermediary_nonlinearities, listsize)
		self.layer_params = []
		last_size = input_size
		curr=1
		for size, init in zip(intermediary_sizes, initializations):
			W,b = init
			self.layer_params.append( (self.create_param(W, (last_size, size), name="Wii%s" % curr), self.create_param(b, (size,), name="bi%s" % curr)))
			last_size = size
			curr += 1

		self.Wio = self.create_param(Wio, (last_size, output_size), name="Wio")
		self.bo = self.create_param(bo, (output_size,), name="bo")
		self.dropout = DropoutPop(p=dropout)

		self.output_nonlinearity=output_nonlinearity


	def get_params(self):
		return [self.Wio] + [param[0] for param in self.layer_params] + self.get_bias_params()

	def get_bias_params(self):
		b = []
		if self.bo is not None:
			b.append(self.bo)

		return b + [param[1] for param in self.layer_params if param[1] is not None]

	def symbolic_call(self, inp, **kwargs):
		"""
		Symbolic call for this ff neural network
		"""
		curr = self.dropout.symbolic_call(inp,**kwargs)
		for param,nonlinearity in zip(self.layer_params, self.intermediary_nonlinearities):
			W,b = param
			intermediate = T.dot(curr, W)
			if b is not None:
				intermediate = intermediate + b.dimshuffle('x',0)
			curr = self.dropout.symbolic_call(nonlinearity(intermediate), **kwargs)

		# now we're at the last intermediate size, so just need to go to output
		output = T.dot(curr, self.Wio)
		if self.bo is not None:
			output = output + self.bo.dimshuffle('x',0)
		return self.output_nonlinearity(output)

