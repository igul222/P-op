import theano
import theano.tensor as T

from .. import nonlinearities
from .. import init
from .. import utils


from .base import Pop
# from lasagne.layers.input import InputLayer
# from lasagne.layers.dense import DenseLayer
# from lasagne.layers import helper
# import custom


class BaseRecurrentPop(Pop):
    """
    My own implementation of a base recurrent Pop (craffel's doesn't allow for easy GRU computation)

    Parameters
    ----------
    incoming: input layer shape batch_size x time_steps x feature_size
    hx_to_h: multiple input layer that takes hidden layer and input layer and outputs the next hidden layer
            ( this is more general than craffel's implementation because it allows for GRU )
    num_units: size of the hidden state
    nonlinearity: TODO: do we need this? could all be done in hx_to_h
    h0: the init function for the initial hidden state
    backwards: whether or not to run this recurrent layer backwards
    learn_init: whether or not to learn h0
    gradient_steps: how far to allow the error to propagate through the sequence (TODO: what is this exactly?)

    My strategy here is to have BaseRecurrentLayer expose two layers, one that outputs the hidden sequence, and one that outputs the input sequences.

    """
    def __init__(self, num_input, num_output, hidden_size,
                 h0=init.Constant(0.), backwards=False,
                 learn_init=False, gradient_steps=-1, **kwargs):
        super(BaseRecurrentPop, self).__init__(num_input,num_output,**kwargs)
        self.num_output=num_output
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.hidden_size=hidden_size
        # Initialize hidden state
        self.h0 = self.create_param(h0, (self.hidden_size,))

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.h0]

    def symbolic_call(self, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable
        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  If None,
                then it assumed that all sequences are of the same length.  If
                not all sequences are of the same length, then it must be
                supplied as a matrix of shape (n_batch, n_time_steps) where
                `mask[i, j] = 1` when `j <= (length of sequence i)` and
                `mask[i, j] = 0` when `j > (length of sequence i)`.
        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        seqs = self.get_seqs(*args)
        non_seqs = self.get_non_seqs(*args)
        outputs_info = self.get_outputs_info(*args)

     
        outputs = utils.make_list(theano.scan(self.step_fn, sequences=seqs, non_sequences=non_seqs,
                             go_backwards=self.backwards,
                             outputs_info=outputs_info,
                             truncate_gradient=self.gradient_steps)[0])

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        outputs = [output.dimshuffle(1,0,2) for output in outputs]

        if self.backwards:
            outputs = [output[:, ::-1, :] for output in outputs]

        if self.num_output==1:
            return outputs[0]
        else:
            return outputs

    def get_outputs_info(self, *args):
        """
        Returns values to be passed in to "outputs_info" of the scan function

        This implements the 'vanilla RNN's outputs info, which is just h0. That's what it usually will be. Assumes X is the first arg.

        Overwrite this.
        """
        X = args[0]
        h0alloc = T.alloc(self.h0, X.shape[0], self.hidden_size)
        return [h0alloc] + self.get_extra_outputs_info(*args[1:])

    def get_extra_outputs_info(self, *args):
        """
        args no longer includes X.

        this is to overwrite.
        """
        return [None for i in range(len(args))]

    def get_seqs(self, *args):
        """
        Returns the values to be passed in to "sequences" of the scan function

        TODO: probably base-case to input
        """
        raise NotImplementedError

    def get_non_seqs(self, *args):
        """
        Returns the values to be passed in to "non_sequences" of the scan function

        TODO: probably base-case to None
        """
        raise NotImplementedError

    def step_fn(self, *args):
        """
        returns the step function. Note that we are here assuming that you will figure out the *args ordering.
        """
        raise NotImplementedError



class GatedRecurrentPop(BaseRecurrentPop):
    """
    Implementation of a Gated Recurrent Layer

    Parameters
    ----------
    incoming: incoming layer shape batch_size x timesteps x feature_size
    **kwargs: args like nonlinearity, h0, backwards, etc. see BaseRecurrentLayer

    """

    def __init__(self, input_size, hidden_size,  Wxh=init.Uniform(), Whh=init.Uniform(), bh=init.Constant(0.), Wxu=init.Uniform(), Wxr=init.Uniform(), Whu=init.Uniform(), Whr=init.Uniform(), bu=init.Constant(0.), br=init.Constant(0.), nonlinearity=nonlinearities.tanh, h0=init.Constant(0.), backwards=False, learn_init=False, gradient_steps=-1, **kwargs):
        super(GatedRecurrentPop,self).__init__(1,1,hidden_size, h0=h0, backwards=backwards, learn_init=learn_init, gradient_steps=gradient_steps, **kwargs)

        self.Wxh=self.create_param(Wxh, (input_size, hidden_size), name="Wxh")
        self.Whh=self.create_param(Whh, (hidden_size, hidden_size), name="Whh")
        self.bh=(self.create_param(bh, (hidden_size,), name="bh") if bh is not None else None)
        self.Wxu=self.create_param(Wxu, (input_size,hidden_size), name="Wxu")
        self.Wxr=self.create_param(Wxr, (input_size,hidden_size), name="Wxr")
        self.Whu=self.create_param(Whu, (hidden_size,hidden_size), name="Whu")
        self.Whr= self.create_param(Whr, (hidden_size,hidden_size), name="Whr")
        self.bu=(self.create_param(bu, (hidden_size,), name="bu") if bu is not None else None)
        self.br=(self.create_param(br, (hidden_size,), name="br") if br is not None else None)

        self.nonlinearity = nonlinearity


    def get_seqs(self, X):
        """
        returns computation that can be down outside of Theano's scan to speed up implementation

        Parameters
        ----------
        inputs: input is x

        REMEMBER THAT WE GET EVERYTHING IN THE WRONG ORDER

        TODO: why not b.dimshuffle('x','x',0)? ez checks w/ theano to see if this is automatically done.
        """
        X = X.dimshuffle(1, 0, 2)
        # we can do all the input dots outside of scan.
        Zx = T.dot(X, self.Wxu) + self.bu.dimshuffle('x', 0)
        Rx = T.dot(X, self.Wxr) + self.br.dimshuffle('x', 0)
        Cx = T.dot(X, self.Wxh) + self.bh.dimshuffle('x',0)
        return [Zx, Rx, Cx]

    def get_non_seqs(self, X):
        return None


    def step_fn(self, *args):
        """
        implements the forward pass of **one timestep** of a GRU recurrent net. This does not implement anything more than one timestep because the BaseRecurrent architecture deals with the steps

        Parameters
        ----------
        inputs: inputs[:-1] comes from get_prescan
                inputs[-1] is batch x hidden size (H_tminus1)

        Returns
        -------
        The final hidden state at this timestep
        """
        zx = args[0]
        rx = args[1]
        cx =  args[2]
        htm1 = args[-1] #batch x hidden
        #update
        z = T.nnet.sigmoid(zx + T.dot(htm1, self.Whu))
        #reset
        r = T.nnet.sigmoid(rx + T.dot(htm1, self.Whr))
        # gated
        rh = r * htm1
        # context
        c = self.nonlinearity(cx + T.dot(rh, self.Whh))
        # actual hidden state
        return ((1 - z) * htm1) + (z * c)

    def get_params(self):
        return [self.Wxh, self.Whh, self.Wxu, self.Wxr, self.Whu, self.Wxr] + self.get_bias_params()

    def get_bias_params(self):
        b = []
        for bias in [self.bh, self.bu, self.br]:
            if bias is not None:
                b.append(bias)
        return b

