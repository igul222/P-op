import pop
import theano
import theano.tensor as T

X = T.scalar()
Y = T.scalar()
dbl = pop.pops.test.DoublePop()
sm = pop.pops.test.SumPop()
chained = sm.chain_pops(dbl, dbl)

out = chained(X,Y)

# === [ NNET ] === #
X = T.matrix() # batch x features
dense1 = pop.pops.dense.DensePop(10,30)
dense2 = pop.pops.dense.DensePop(30, 10)
softmax = pop.pops.dense.DensePop(10,3,nonlinearity=pop.nonlinearities.softmax)
chained1 = softmax.chain_pops(dense2)
chained2=chained1.chain_pops(dense1)

out = chained2(X)
fn = theano.function([X], out)


chained = dense1 + dense2 + softmax
out = chained(X)
fn = theano.function([X], out)

