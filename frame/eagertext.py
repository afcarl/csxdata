import warnings

import numpy as np
from parser import parser

from .abstract_frame import Frame
from ..features import embedding_factory
from ..utilities.const import log


class BasicSequence(Frame):

    type = "basicsequence"

    def __init__(self, source, time, cross_val=0.2, **kw):
        if isinstance(source, str):
            source = parser.txt(source, ngram=kw.pop("ngram", 1))
        X, Y = [], []
        for start in range(0, len(source)-time-2):
            X.append(source[start:start+time])
            Y.append(source[start+time+1])
        super().__init__(
            (np.array(X), np.array(Y)), cross_val, indeps=0, headers=None, **kw
        )

    def reset_data(self, shuff, transform, trparam=None):
        pass

    @property
    def neurons_required(self):
        pass

    def concatenate(self, other):
        pass


class EagerText(Frame):

    type = "sequence"

    def __init__(self, source, embeddim=None, cross_val=0.2, n_gram=1, timestep=None, **parser_kw):

        def split_X_y(dat):
            d = []
            dp = []
            if timestep:
                start = 0
                end = timestep
                while end <= dat.shape[0]:
                    slc = dat[start:end]
                    d.append(slc[:-1])
                    dp.append(slc[-1])
                    start += 1
                    end += 1
                d = np.stack(d)
                dp = np.stack(dp)
            else:
                d = np.array([[e for e in time[:-1]] for time in dat])
                dp = np.array([time[-1] for time in dat])
            return d, dp

        self._embedding = None
        self.timestep = timestep

        chararr = parser.txt(source, ngram=n_gram, **parser_kw)
        self._embedding = embedding_factory(embeddim).fit(chararr)
        data = self._embedding(chararr)
        data, deps = split_X_y(data)

        super().__init__((data, deps), cross_val=cross_val, indeps=0, headers=None, **parser_kw)

        self.reset_data(shuff=True)

    def reset_data(self, shuff: bool, transform=None, trparam: int=None):
        if transform is not None:
            transform = None
        Frame.reset_data(self, shuff=shuff, transform=transform)

    @property
    def neurons_required(self):
        return (self.timestep - 1, self._embedding.dim), (self._embedding.dim,)

    def translate(self, preds, use_proba=False):
        return _translate(preds, use_proba, self._embedding)

    def primer(self):
        from random import randrange
        primer = self._learning[randrange(self.N)]
        return primer.reshape(1, *primer.shape)

    def concatenate(self, other):
        pass


class LazyText:

    def __init__(self, source, embeddim=None, cross_val=0.2, n_gram=1, timestep=None, coding="utf-8-sig"):

        def chop_up_to_timesteps():
            newN = self.data.shape[0] // timestep
            if self.data.shape[0] % timestep != 0:
                warnings.warn("Trimming data to fit into timesteps!", RuntimeWarning)
                self.data = self.data[:self.data.shape[0] - (self.data.shape[0] % timestep)]
            newshape = newN, timestep
            print("Reshaping from: {} to: {}".format(self.data.shape, newshape))
            self.data = self.data.reshape(*newshape)

        self.timestep = timestep
        self._crossval = cross_val

        self.data = np.ravel(parser.txt(source, ngram=n_gram, coding=coding))
        self._embedding = embedding_factory(embeddim).fit(self.data)
        chop_up_to_timesteps()

        self.n_testing = int(self.data.shape[0] * cross_val)
        self.N = self.data.shape[0]

    @property
    def neurons_required(self):
        return (self.timestep - 1, self._embedding.dim), self._embedding.dim

    def translate(self, preds, use_proba=False):
        return _translate(preds, use_proba, self._embedding)

    def batchgen(self, bsize):
        index = 0
        epochs_passed = 0
        while 1:
            start = bsize * index
            end = start + bsize

            slc = self.data[start:end]
            slc = self._embedding(slc)

            X, y = slc[:, :-1, :], slc[:, -1]

            if end > self.N:
                warnings.warn("\nEPOCH PASSED!", RuntimeWarning)
                epochs_passed += 1
                log("{} MASSIVE_SEQUENCE EPOCHS PASSED!".format(epochs_passed))

                index = 0
            index += 1

            yield X, y

    def primer(self):
        from random import randrange
        primer = self.data[randrange(self.N)]
        primer = self._embedding(primer)
        return primer.reshape(1, *primer.shape)

    def concatenate(self, other):
        pass


class WordSequence:

    def __init__(self, source, embeddim=None, cross_val=0.2, **kw):
        from parser.reparse import reparse_txt

        source = reparse_txt(source, **kw)

        chars = set(parser.txt(source, ngram=1))
        words = source.replace("\n", " ").split(" ")

        self._embedding = embedding_factory(embeddim).fit(chars)
        self.data = np.array([np.array([self._embedding(c) for c in w]) for w in words])

        self.n_testing = int(self.data.shape[0] * cross_val)
        self.N = self.data.shape[0]

    def table(self, *arg, **kw):
        raise NotImplementedError

    @property
    def neurons_required(self):
        return (self._embedding.dim,),  (self._embedding.dim,)

    def translate(self, preds, use_proba=False):
        return _translate(preds, use_proba, self._embedding)

    def batchgen(self, bsize, *args, **kw):
        index = 0
        epochs_passed = 0
        while 1:
            start = bsize * index
            end = start + bsize

            slc = self.data[start:end]
            slc = self._embedding(slc)

            X, y = slc[:, :-1, :], slc[:, -1]

            index += 1
            if end > self.N:
                warnings.warn("\nEPOCH PASSED!", RuntimeWarning)
                epochs_passed += 1
                log("{} MASSIVE_SEQUENCE EPOCHS PASSED!".format(epochs_passed))

                index = 0

            yield X, y


def _translate(preds, use_proba, embedding):
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]
    elif preds.ndim == 2:
        pass
    else:
        raise NotImplementedError("Oh-oh...")

    if use_proba:
        preds = np.log(preds)
        e_preds = np.exp(preds)
        preds = e_preds / e_preds.sum()
        preds = np.random.multinomial(1, preds, size=preds.shape)

    human = embedding.translate(preds)
    return human
