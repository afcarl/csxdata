from random import randrange

import numpy as np

from csxdata.frames import Sequence
from keras.models import Sequential


def speak_to_me(net: Sequential, dat: Sequence):
    chain = ""
    pred = dat.learning[randrange(dat.N)]
    for _ in range(100):
        np.row_stack((pred[-4:], net.predict(pred)))
        human = dat._embedding.translate(pred[-1])
        print(human, end="")
        chain += human
    print()


