import numpy as np

from frame.sequence import Sequence


def speak_to_me(net, dat: Sequence, stochastic=False, ngrams=50):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    for _ in range(ngrams):
        inputs = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.prediction(inputs)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain


def keras_speak(net, dat, stochastic=False, ngrams=50):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    for _ in range(ngrams):
        inputs = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.predict(inputs)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain
