import numpy as np

import theano
import cPickle as pkl
from .. import utils


__all__ = [
    "Pop",
    "IdentityPop",
    "Layer",
    "MultipleInputsLayer",
    "RecurrentComputationLayer",
]


# Layer base class

class Pop(object):
    """
    The base class P-op class represents a parameterized theano op.

    It is subclassed when implementing new parametrized Ops.
    """
    def __init__(self, num_input, num_output, name=None):
        """
        Initializes the Pop

        Parameters
        ----------
        num_input: The number of input this Pop takes
        num_outputs: The number of outputs this Pop returns
        name: optional name
        """
        self.num_input=num_input
        self.num_output=num_output
        self.name=name
        self.Pops = [self]
        self.Pop_mapping = {}
        if name is not None:
            self.Pop_mapping[name] = self

    def get_output(self, *args, **kwargs):
        """
        This currently just exists to fit Lasagne's objectives/etc. TODO: keep or remove. why "symbolic call"?
        """
        return self.symbolic_call(*args, **kwargs)

    def chain_pops(self, *pops, **kwargs):
        """
        Returns a new Pop that represents the chain of this Pop taking inputs as the outputs of *pops.

        Parameters
        ----------
        *pops: the Pops that we want to consider as inputs

        TODO: allow for some pops to not be known, i.e partials.
        TODO: look into python wrappers for the callable stuff
        """
        # check that the sizes match
        pop_outputs = [pop.num_output for pop in pops]
        if not sum(pop_outputs) == self.num_input:
            raise ValueError("You tried to chain Pops with an incorrect number of inputs")

        # create new Pop
        # write name
        name = kwargs.get('name', None)
        if name is None:
            name = '_'.join([pop.name if pop.name is not None else 'unk'for pop in pops])

        # how many inputs does it take?
        pop_inputs = [pop.num_input for pop in pops]
        num_inputs = sum(pop_inputs)

        # initialize new Pop
        new_Pop = Pop(num_input=num_inputs, num_output=self.num_output, name=name)
        # add all Pops together to keep track
        new_Pop.Pops += sum([pop.Pops for pop in pops], [])
        new_Pop.Pops += self.Pops

        # fill the Pop mapping
        new_Pop.Pop_mapping = self.Pop_mapping
        for pop in pops:
            for name in pop.Pop_mapping:
                mapped = pop.Pop_mapping[name]
                if name in new_Pop.Pop_mapping:
                    # see if the mapped is the same
                    if mapped == new_Pop.Pop_mapping[name]:
                        continue
                    else:
                        raise ValueError("Same Pop name, different Pop. TODO: add support for this somehow")
                else:
                    new_Pop.Pop_mapping[name] = mapped

        # make new callable
        # todo: have length of args checks somewhere
        # todo: pass **kwargs somehow
        # possible todo: maybe somehow allow giving args by name, i.e pop1arg=__, pop2arg=___.r
        def new_symbolic_call(*args, **ckwargs):
            outputs = []
            curr_offset = 0
            for pop in pops:
                out = pop.symbolic_call(*(args[curr_offset:curr_offset+pop.num_input]), **ckwargs)
                outputs += utils.make_list(out)
                curr_offset += pop.num_input
            # now outputs should have enough to call this pop's symbolic_call
            final_out = self.symbolic_call(*outputs, **ckwargs)
            return final_out

        new_Pop.symbolic_call = new_symbolic_call

        return new_Pop

    def __add__(self, other):
        """
        Shortcut to chain Pops
        """
        return other.chain_pops(self)

    def __call__(self, *args):
        return self.symbolic_call(*args)

    def get_all_params(self, ignore=[], include=[]):
        """
        Gets all the params of this entire Pop, ignoring the Pops listed in ignore.

        Parameters
        ----------
        ignore: a list of Pops to ignore when getting parameters

        TODO: is 'ignore' better or is 'include' better?
        TODO: should we deal with biases seperately, maybe?
        """
        # if include is 0-length, include it all
        if len(include) == 0:
            include = self.Pops

        # now return the ones not in ignore
        return utils.unique(sum([pop.get_params() for pop in include if pop not in ignore], []))

    def get_params(self):
        """
        Gets the params of this specific Pop.
        """
        return []

    def create_param(self, param, shape, name=None):
        """
        Helper method to create Theano shared variables for layer parameters
        and to initialize them.

        :parameters:
            - param : numpy array, Theano shared variable, or callable
                One of three things:
                    * a numpy array with the initial parameter values
                    * a Theano shared variable representing the parameters
                    * a function or callable that takes the desired shape of
                      the parameter array as its single argument.

            - shape : tuple
                a tuple of integers representing the desired shape of the
                parameter array.

        :returns:
            - variable : Theano shared variable
                a Theano shared variable representing layer parameters. If a
                numpy array was provided, the variable is initialized to
                contain this array. If a shared variable was provided, it is
                simply returned. If a callable was provided, it is called, and
                its output is used to initialize the variable.

        :note:
            This method should be used in `__init__()` when creating a
            :class:`Layer` subclass that has trainable parameters. This
            enables the layer to support initialization with numpy arrays,
            existing Theano shared variables, and callables for generating
            initial parameter values.
        """
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        if isinstance(param, theano.compile.SharedVariable):
            # We cannot check the shape here, the shared variable might not be
            # initialized correctly yet. We can check the dimensionality
            # though. Note that we cannot assign a name here.
            if param.ndim != len(shape):
                raise RuntimeError("shared variable has %d dimensions, "
                                   "should be %d" % (param.ndim, len(shape)))
            return param

        elif isinstance(param, np.ndarray):
            if param.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (param.shape, shape))
            return theano.shared(param, name=name)

        elif hasattr(param, '__call__'):
            arr = param(shape)
            if not isinstance(arr, np.ndarray):
                raise RuntimeError("cannot initialize parameters: the "
                                   "provided callable did not return a numpy "
                                   "array")

            return theano.shared(utils.floatX(arr), name=name)

        elif param is None:
            return None

        else:
            raise RuntimeError("cannot initialize parameters: 'param' is not "
                               "a numpy array, a Theano shared variable, or a "
                               "callable")


    def serialize(self, filename):
        """
        Serializes this Pop into filename

        Can't just pickle the Pop because of the new_symbolic_call stuff
        """
        params = [param.get_value() for param in self.get_all_params()]
        pkl.dump(params, open(filename,'w'))

    def load(self, filename):
        """
        loads the parameters from filename
        """
        params = self.get_all_params()
        loaded = pkl.load(open(filename))
        for param,lparam in zip(params, loaded):
            param.set_value(lparam.astype(theano.config.floatX))


    def symbolic_call(self, *args, **kwargs):
        """
        Represents the function of this Pop. Accepts and returns a symbolic theano variable.

        Parameters
        ----------
        *args: the symbolic theano variable to use as input
        """
        raise NotImplementedError

    

class IdentityPop(Pop):
    """
    This is a placeholder Pop that takes in a symbolic variable and just returns it.

    In the future, this should be removed, and chain_pops should just allow you to pass in a symbolic variable
    itself rather than forcing them to be Pops.
    """
    def __init__(self, **kwargs):
        super(IdentityPop, self).__init__(1,1,**kwargs)

    def symbolic_call(self, inp, **kwargs):
        return inp



class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network.
    It should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the
    full network.
    """
    def __init__(self, incoming, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incoming : a :class:`Layer` instance or a tuple
                the layer feeding into this layer, or the expected input shape
            - name : a string or None
                an optional name to attach to this layer
        """
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.get_output_shape()
            self.input_layer = incoming
        self.name = name

    def get_params(self):
        """
        Returns a list of all the Theano variables that parameterize the
        layer.

        :returns:
            - list
                the list of Theano variables.

        :note:
            By default this returns an empty list, but it should be overridden
            in a subclass that has trainable parameters.
        """
        return []

    def get_bias_params(self):
        """
        Returns a list of all the Theano variables that are bias parameters
        for the layer.

        :returns:
            - bias_params : list
                the list of Theano variables.

        :note:
            By default this returns an empty list, but it should be overridden
            in a subclass that has trainable parameters.

            While `get_params()` should return all Theano variables,
            `get_bias_params()` should only return those corresponding to bias
            parameters. This is useful when specifying regularization (it is
            often undesirable to regularize bias parameters).
        """
        return []

    def get_output_shape(self):
        """
        Computes the output shape of the network at this layer.

        :returns:
            - output shape: tuple
                a tuple that represents the output shape of this layer. The
                tuple has as many elements as there are output dimensions, and
                the elements of the tuple are either integers or `None`.

        :note:
            When implementing a new :class:`Layer` class, you will usually
            keep this unchanged and just override `get_output_shape_for()`.
        """
        return self.get_output_shape_for(self.input_shape)

    def get_output(self, input=None, **kwargs):
        """
        Computes the output of the network at this layer. Optionally, you can
        define an input to propagate through the network instead of using the
        input variables associated with the network's input layers.

        :parameters:
            - input : None, Theano expression, numpy array, or dict
                If None, uses the inputs of the :class:`InputLayer` instances.
                If a Theano expression, this will replace the inputs of all
                :class:`InputLayer` instances (useful if your network has a
                single input layer).
                If a numpy array, this will be wrapped as a Theano constant
                and used just like a Theano expression.
                If a dictionary, any :class:`Layer` instance (including the
                input layers) can be mapped to a Theano expression or numpy
                array to use instead of its regular output.

        :returns:
            - output : Theano expression
                the output of this layer given the input to the network

        :note:
            When implementing a new :class:`Layer` class, you will usually
            keep this unchanged and just override `get_output_for()`.
        """
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        elif self.input_layer is None:
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its input from. "
                               "Did you mean get_output_for()?")
        else:  # in all other cases, just pass the input on to the next layer.
            layer_input = self.input_layer.get_output(input, **kwargs)
            return self.get_output_for(layer_input, **kwargs)

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        :parameters:
            - input_shape : tuple
                a tuple representing the shape of the input. The tuple should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method will typically be overridden when implementing a new
            :class:`Layer` class. By default it simply returns the input
            shape. This means that a layer that does not modify the shape
            (e.g. because it applies an elementwise operation) does not need
            to override this method.
        """
        return input_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        :parameters:
            - input : Theano expression
                the expression to propagate through this layer

        :returns:
            - output : Theano expression
                the output of this layer given the input to this layer

        :note:
            This is called by the base :class:`Layer` implementation to
            propagate data through a network in `get_output()`. While
            `get_output()` asks the underlying layers for input and thus
            returns an expression for a layer's output in terms of the
            network's input, `get_output_for()` just performs a single step
            and returns an expression for a layer's output in terms of
            that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def create_param(self, param, shape, name=None):
        """
        Helper method to create Theano shared variables for layer parameters
        and to initialize them.

        :parameters:
            - param : numpy array, Theano shared variable, or callable
                One of three things:
                    * a numpy array with the initial parameter values
                    * a Theano shared variable representing the parameters
                    * a function or callable that takes the desired shape of
                      the parameter array as its single argument.

            - shape : tuple
                a tuple of integers representing the desired shape of the
                parameter array.

        :returns:
            - variable : Theano shared variable
                a Theano shared variable representing layer parameters. If a
                numpy array was provided, the variable is initialized to
                contain this array. If a shared variable was provided, it is
                simply returned. If a callable was provided, it is called, and
                its output is used to initialize the variable.

        :note:
            This method should be used in `__init__()` when creating a
            :class:`Layer` subclass that has trainable parameters. This
            enables the layer to support initialization with numpy arrays,
            existing Theano shared variables, and callables for generating
            initial parameter values.
        """
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        if isinstance(param, theano.compile.SharedVariable):
            # We cannot check the shape here, the shared variable might not be
            # initialized correctly yet. We can check the dimensionality
            # though. Note that we cannot assign a name here.
            if param.ndim != len(shape):
                raise RuntimeError("shared variable has %d dimensions, "
                                   "should be %d" % (param.ndim, len(shape)))
            return param

        elif isinstance(param, np.ndarray):
            if param.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (param.shape, shape))
            return theano.shared(param, name=name)

        elif hasattr(param, '__call__'):
            arr = param(shape)
            if not isinstance(arr, np.ndarray):
                raise RuntimeError("cannot initialize parameters: the "
                                   "provided callable did not return a numpy "
                                   "array")

            return theano.shared(utils.floatX(arr), name=name)

        else:
            raise RuntimeError("cannot initialize parameters: 'param' is not "
                               "a numpy array, a Theano shared variable, or a "
                               "callable")


class MultipleInputsLayer(Layer):
    """
    This class represents a layer that aggregates input from multiple layers.
    It should be subclassed when implementing new types of layers that
    obtain their input from multiple layers.
    """
    def __init__(self, incomings, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incomings : a list of :class:`Layer` instances or tuples
                the layers feeding into this layer, or expected input shapes
            - name : a string or None
                an optional name to attach to this layer
        """
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.get_output_shape()
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name

    def get_output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output(self, input=None, **kwargs):
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        elif any(input_layer is None for input_layer in self.input_layers):
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its inputs from. "
                               "Did you mean get_output_for()?")
        # In all other cases, just pass the network input on to the next layers
        else:
            layer_inputs = [input_layer.get_output(input, **kwargs) for
                            input_layer in self.input_layers]
            return self.get_output_for(layer_inputs, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.

        :parameters:
            - input_shape : list of tuple
                a list of tuples, with each tuple representing the shape of
                one of the inputs (in the correct order). These tuples should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method must be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).

        :parameters:
            - inputs : list of Theano expressions
                The Theano expressions to propagate through this layer

        :returns:
            - output : Theano expressions
                the output of this layer given the inputs to this layer

        :note:
            This is called by the base :class:`MultipleInputsLayer`
            implementation to propagate data through a network in
            `get_output()`. While `get_output()` asks the underlying layers
            for input and thus returns an expression for a layer's output in
            terms of the network's input, `get_output_for()` just performs a
            single step and returns an expression for a layer's output in
            terms of that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

class MultipleSplitInputsLayer(Layer):
    """
    This class is just a re-writing of Lasagne's multiple inputs layer that allows you to send a list of inputs into the layer and have them be split up accordingly

    TODO: clean up code/docs and possibly add support for variable-lenght splitting, i.e incomings will take #inputs that it takes. this should be ez to do and nice.
    TODO: move to lasagne fork for readability & standardization
    """
    def __init__(self, incomings, name=None):
        """
        Instantiates the layer.
        :parameters:
            - incomings : a list of :class:`Layer` instances or tuples
                the layers feeding into this layer, or expected input shapes
            - name : a string or None
                an optional name to attach to this layer
        """
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.get_output_shape()
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name

    def get_output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output(self, inputs=None, **kwargs):
        if isinstance(inputs, dict) and (self in inputs):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(inputs[self])
        elif any(input_layer is None for input_layer in self.input_layers):
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its inputs from. "
                               "Did you mean get_output_for()?")
        # In all other cases, just pass the network input on to the next layers
        else:
            layer_inputs = [input_layer.get_output(input, **kwargs) for
                            input_layer,input in zip(self.input_layers, inputs)]
            return self.get_output_for(layer_inputs, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.
        :parameters:
            - input_shape : list of tuple
                a list of tuples, with each tuple representing the shape of
                one of the inputs (in the correct order). These tuples should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.
        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.
        :note:
            This method must be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).
        :parameters:
            - inputs : list of Theano expressions
                The Theano expressions to propagate through this layer
        :returns:
            - output : Theano expressions
                the output of this layer given the inputs to this layer
        :note:
            This is called by the base :class:`MultipleInputsLayer`
            implementation to propagate data through a network in
            `get_output()`. While `get_output()` asks the underlying layers
            for input and thus returns an expression for a layer's output in
            terms of the network's input, `get_output_for()` just performs a
            single step and returns an expression for a layer's output in
            terms of that layer's input.
            This method should be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

"""
todo: basecomputationlayer? needed?
"""
class RecurrentComputationLayer(Layer):
    """
    This class represents a layer that computes a recurrent layer's function.
    It should be subclassed when implementing new types of layers that compute recurrent outputs,
    e.g LSTM/GRU. See my GRU implementation.
    """
    def __init__(self, name=None):
        """
        Instantiates the layer.

        No need for any input layer as this is just a computation block.

        :parameters:
            - name : a string or None
                an optional name to attach to this layer
        """
        self.name=name


    def get_output_shape(self):
        """
        probably not needed
        """
        return self.get_output_shape_for(self.input_shape)

    def get_output(self, input=None, **kwargs):
        """
        This should never be used.
        """
        raise Exception("You tried to use get_output on a recurrent computation layer.")
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        elif any(input_layer is None for input_layer in self.input_layers):
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its inputs from. "
                               "Did you mean get_output_for()?")
        # In all other cases, just pass the network input on to the next layers
        else:
            layer_inputs = [input_layer.get_output(input, **kwargs) for
                            input_layer in self.input_layers]
            return self.get_output_for(layer_inputs, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.

        :parameters:
            - input_shape : list of tuple
                a list of tuples, with each tuple representing the shape of
                one of the inputs (in the correct order). These tuples should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method must be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).

        :parameters:
            - inputs : list of Theano expressions
                The Theano expressions to propagate through this layer
                inputs[0] is expected to be the input of the recurrent net;
                inputs[1:] is expected to be from the past timesteps (usually, inputs will just be 2-d, so this is just H[t-1])

        :returns:
            - output : Theano expressions
                the output of this layer given the inputs to this layer

        :note:
        TODO**:**:**
            This is called by the base :class:`RecurrentComputationLayer`
            implementation to propagate data through a network in
            `get_output()`. While `get_output()` asks the underlying layers
            for input and thus returns an expression for a layer's output in
            terms of the network's input, `get_output_for()` just performs a
            single step and returns an expression for a layer's output in
            terms of that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError
