import numpy as np

from csxdata.frames import Sequence
from keras.models import Sequential


def speak_to_me(net: Sequential, dat: Sequence):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    print("Generating with primer: {}".format(chain))
    for _ in range(120):
        ingoes = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.predict(ingoes)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred)
        chain += "".join(human)
    return chain
