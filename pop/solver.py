"""
Solver class for P-op models
ankitk@stanford.edu

This is a huge work-in-progress. It should be combined with a dataset class.
"""


class Solver(object):
    """
    The base Solver class.
    """
    def __init__(self, updates, name=None):
        """
        updates is the P-op updates function
        """
        self.name = name
        self.updates = updates


    def solve(model, loss_function, update_kwargs, X_vars, y_var, output, train_iterable, num_epochs):
        """
        main solve function
        """
        obj = pop.objectives.Objective(model, loss_function=loss_function)
        loss = obj.get_loss(*X_vars, target=y_var, output=output)

        # get regularization, learn params
        reg_params = model.get_all_params(include=reg_include, ignore=reg_ignore)
        print "Regularized params:"
        print reg_params
        reg = sum(T.sum(p**2) for p in reg_params) # l2 hardcoded
        if rho == 0:
            print "Rho was 0, not using regularization"
        else:
            loss = loss + rho*reg

        # learn params
        params = params.get_all_params(include=include, ignore=ignore)
        print "Learned params:"
        print params

        updates = self.updates(loss, params, *update_kwargs)
        train = theano.function(X_vars + [y_var], loss, updates=updates)
        predict = theano.function(X_vars)

