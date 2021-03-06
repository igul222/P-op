import theano
import theano.tensor as T
import functools
import numpy as np
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
                 learn_init=False, gradient_steps=-1, bidirectional=False, flip_backwards=False, **kwargs):
        super(BaseRecurrentPop, self).__init__(num_input,num_output,**kwargs)
        self.num_output=num_output
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.bidirectional = bidirectional
        self.hidden_size=hidden_size
        self.flip_backwards=flip_backwards
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
        seqs = self.get_seqs(*args, **kwargs)
        if len(seqs) == 0:
            n_steps = self.get_n_steps(*args, **kwargs)
        else:
            n_steps = None
        non_seqs = self.get_non_seqs(*args, **kwargs)
        outputs_info = self.get_outputs_info(*args, **kwargs)


        if self.bidirectional:
            outputs_forward=utils.make_list(theano.scan(functools.partial(self.step_fn, **kwargs), sequences=seqs, non_sequences=non_seqs,
                             go_backwards=False,
                             outputs_info=outputs_info,
                             truncate_gradient=self.gradient_steps)[0])
            outputs_backward=utils.make_list(theano.scan(functools.partial(self.step_fn, **kwargs), sequences=seqs, non_sequences=non_seqs,
                             go_backwards=True,
                             outputs_info=outputs_info,
                             truncate_gradient=self.gradient_steps)[0])

            # dimshuffle back
            outputs_forward = [output.dimshuffle(1,0,2) for output in outputs_forward]
            outputs_backward = [output.dimshuffle(1,0,2) for output in outputs_backward]

            # flip backwards runs
            outputs_backward = [output[:,::-1,:] for output in outputs_backward]

            # and concatenate
            outputs = [T.concatenate([forward,backward],axis=2) for forward,backward in zip(outputs_forward,outputs_backward)]

        else:
            outputs, upd = theano.scan(functools.partial(self.step_fn, **kwargs), sequences=seqs, n_steps=n_steps, non_sequences=non_seqs,
                                 go_backwards=self.backwards,
                                 outputs_info=outputs_info,
                                 truncate_gradient=self.gradient_steps)

            outputs = utils.make_list(outputs)
            self.updates =upd
            print 'updates:'
            print len(upd)
            # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
            outputs = [output.dimshuffle(1,0,2) for output in outputs]

            if self.flip_backwards:
                outputs = [output[:, ::-1, :] for output in outputs]

        if self.num_output==1:
            return outputs[0]
        else:
            return outputs

    def get_outputs_info(self, *args, **kwargs):
        """
        Returns values to be passed in to "outputs_info" of the scan function

        This implements the 'vanilla RNN's outputs info, which is just h0. That's what it usually will be. Assumes X is the first arg.

        Overwrite this.
        """
        X = args[0]
        h0alloc = T.alloc(self.h0, X.shape[0], self.hidden_size)
       
        h0alloc = T.unbroadcast(h0alloc, 0)
       
        return [h0alloc] + self.get_extra_outputs_info(*args)

    def get_extra_outputs_info(self, *args, **kwargs):
        """
        args no longer includes X.

        this is to overwrite.
        """
        return []

    def get_function_updates(self, *args, **kwargs):
        return self.updates

    def get_seqs(self, *args, **kwargs):
        """
        Returns the values to be passed in to "sequences" of the scan function

        TODO: probably base-case to input
        """
        raise NotImplementedError

    def get_non_seqs(self, *args, **kwargs):
        """
        Returns the values to be passed in to "non_sequences" of the scan function

        TODO: probably base-case to None
        """
        raise NotImplementedError

    def step_fn(self, *args, **kwargs):
        """
        returns the step function. Note that we are here assuming that you will figure out the *args ordering.
        """
        raise NotImplementedError

    def get_n_steps(self, *args, **kwargs):
        """
        returns number of steps to take. only called if get_seqs returns an empty list
        """
        raise NotImplementedError




class BaseSequenceGeneratorPop(Pop):
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
                 h0=init.Constant(0.), p0=init.Constant(0.), backwards=False,
                 learn_h0=False, learn_p0=True, gradient_steps=-1, **kwargs):
        super(BaseSequenceGeneratorPop, self).__init__(num_input,num_output,**kwargs)
        self.num_output=num_output
        self.learn_h0=learn_h0
        self.learn_p0=learn_p0
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.hidden_size=hidden_size
        # Initialize hidden state
        self.h0 = self.create_param(h0, (self.hidden_size,))
        self.p0 = self.create_param(p0, (self.hidden_size,))


    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        p = []
        if self.learn_h0:
            p.append(self.h0)

        if self.learn_p0:
            p.append(self.p0)

        return p

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
        # dimshuffling in get_seqs if necessary.
        seqs = self.get_seqs(*args, **kwargs)
        if len(seqs) == 0:
            n_steps = self.get_n_steps(*args, **kwargs)
        else:
            n_steps = None
        non_seqs = self.get_non_seqs(*args, **kwargs)
        outputs_info = self.get_outputs_info(*args, **kwargs)


        outputs, upd = theano.scan(functools.partial(self.step_fn, **kwargs), sequences=seqs, n_steps=n_steps, non_sequences=non_seqs,
                             go_backwards=self.backwards,
                             outputs_info=outputs_info,
                             truncate_gradient=self.gradient_steps)

        outputs = utils.make_list(outputs)
        self.updates =upd

        if self.num_output==1:
            return outputs[0]
        else:
            return outputs

    def get_outputs_info(self, *args, **kwargs):
        """
        Returns values to be passed in to "outputs_info" of the scan function

        This implements the 'vanilla RNN's outputs info, which is just h0. That's what it usually will be. Assumes X is the first arg.

        Overwrite this.
        """
        # print T.unbroadcast(self.h0[np.newaxis,:],1).type
        # print T.unbroadcast(self.h0[np.newaxis,:],1).broadcast
        return [T.unbroadcast(self.h0[np.newaxis,:],0), T.unbroadcast(self.p0[np.newaxis,:],0)] + self.get_extra_outputs_info(*args, **kwargs)

    def get_extra_outputs_info(self, *args, **kwargs):
        """
        args no longer includes X.

        this is to overwrite.
        """
        return []

    def get_function_updates(self, *args, **kwargs):
        return self.updates

    def get_seqs(self, *args, **kwargs):
        """
        Returns the values to be passed in to "sequences" of the scan function

        TODO: probably base-case to input
        """
        raise NotImplementedError

    def get_non_seqs(self, *args, **kwargs):
        """
        Returns the values to be passed in to "non_sequences" of the scan function

        TODO: probably base-case to None
        """
        raise NotImplementedError

    def step_fn(self, *args, **kwargs):
        """
        returns the step function. Note that we are here assuming that you will figure out the *args ordering.
        """
        raise NotImplementedError

    def get_n_steps(self, *args, **kwargs):
        """
        returns number of steps to take. only called if get_seqs returns an empty list
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


    def get_seqs(self, X, **kwargs):
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

    def get_non_seqs(self, X, **kwargs):
        return None


    def step_fn(self, *args, **kwargs):
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

class VanillaRecurrent(BaseRecurrentPop):
    """
    Implementation of a Vanilla Recurrent Layer
    """

    def __init__(self, input_size, hidden_size,  Wxh=init.Uniform(), Whh=init.Uniform(), bh=init.Constant(0.), nonlinearity=nonlinearities.tanh, h0=init.Constant(0.), backwards=False, learn_init=False, gradient_steps=-1, **kwargs):
        super(GatedRecurrentPop,self).__init__(1,1,hidden_size, h0=h0, backwards=backwards, learn_init=learn_init, gradient_steps=gradient_steps, **kwargs)

        self.Wxh=self.create_param(Wxh, (input_size, hidden_size), name="Wxh")
        self.Whh=self.create_param(Whh, (hidden_size, hidden_size), name="Whh")
        self.bh=(self.create_param(bh, (hidden_size,), name="bh") if bh is not None else None)

        self.nonlinearity = nonlinearity


    def get_seqs(self, X, **kwargs):
        """
        returns computation that can be down outside of Theano's scan to speed up implementation

        Parameters
        ----------
        inputs: input is x

        REMEMBER THAT WE GET EVERYTHING IN THE WRONG ORDER

        TODO: why not b.dimshuffle('x','x',0)? ez checks w/ theano to see if this is automatically done.
        """
        X = X.dimshuffle(1, 0, 2)
        prop = T.dot(X, self.Wxh) + self.bh.dimshuffle('x',0)
        return [prop]

    def get_non_seqs(self, X, **kwargs):
        return None


    def step_fn(self, *args, **kwargs):
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
        xprop = args[0]
        htm1 = args[-1] #batch x hidden
        return self.nonlinearity(T.dot(htm1, self.Whh) + xprop)
    def get_params(self):
        return [self.Wxh, self.Whh] + self.get_bias_params()

    def get_bias_params(self):
        b = []
        for bias in [self.bh]:
            if bias is not None:
                b.append(bias)
        return b


