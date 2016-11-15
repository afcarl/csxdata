import numpy as np

from csxdata.frames import Sequence


def speak_to_me(net, dat: Sequence, stochastic=False, ngrams=50):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    for _ in range(ngrams):
        ingoes = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.predict(ingoes)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain
