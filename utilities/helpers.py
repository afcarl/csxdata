import numpy as np

from csxdata.frames import Sequence
from keras.models import Sequential


def speak_to_me(net: Sequential, dat: Sequence, stochastic=False, ngrams=50):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    print("Generating with primer: {}".format(chain))
    for _ in range(ngrams):
        ingoes = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.predict(ingoes)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain
