import warnings

import numpy as np
from .abstract_frame import Frame
from ..utilities.vectorop import shuffle
from ..features import embedding_factory


class CData(Frame):
    """
    Class is for holding categorical learning data
    """

    def __init__(self, source, indeps=1, headers=0, cross_val=.2, **kw):
        Frame.__init__(self, source, cross_val, indeps, headers, **kw)

        if type(self.indeps[0]) in (np.ndarray, tuple, list):
            self.indeps = np.array([d[0] for d in self.indeps])

        self.type = "classification"
        self._embedding = None

        transformation = kw.get("transformation")
        parameter = kw.get("traparam")
        embedding = kw.get("embedding", 0)

        self.reset_data(shuff=False, embedding=embedding, transform=transformation, trparam=parameter)

    def reset_data(self, shuff=True, embedding: int=0, transform=None, trparam=None):
        """
        Regenerates learning and testing from the original data.

        :param shuff: whether data should be shuffled
        :param embedding: dimensionality of enbedding (0 means one-hot approach)
        :param transform: name of transfromation or True to keep the current one
        :param trparam: parameter supplied for transformation
        """
        Frame.reset_data(self, shuff, transform, trparam)
        self.embedding = embedding

    @property
    def embedding(self):
        return self._embedding.name

    @embedding.setter
    def embedding(self, emb):
        """
        Sets the embedding of the dependent variable (Y)

        :param emb: dimensionality of embedding (0 means one-hot)
        """
        self._embedding = embedding_factory(emb).fit(self.indeps)

    @embedding.deleter
    def embedding(self):
        """Resets any previous embedding to one-hot"""
        self.embedding = 0

    def embed(self, Y):
        return self._embedding(Y)

    def batchgen(self, bsize: int, data: str="learning", weigh=False, infinite=False):
        """
        Returns a generator which yields batches from the data

        :param bsize: specifies the size of the batches yielded
        :param data: string specifing the data subset (learning/testing)
        :param weigh: whether to yield sample weights as well
        :param infinite: if set, the iteration wraps around.
        """
        tab = self.table(data, weigh=weigh)
        n = len(tab[0])
        start = 0
        end = start + bsize

        while 1:
            if end >= n:
                end = n
            if start >= n:
                if infinite:
                    start = 0
                    end = start + bsize
                    tab = shuffle(tab)
                else:
                    break

            yield tuple(map(lambda elem: elem[start:end], tab))

            start += bsize
            end += bsize

    def table(self, data="learning", shuff=True, m=None, weigh=False):
        """
        Returns a learning table, a tuple of (X, Y[, w])

        :param data: specifies the data subset to return
        :param shuff: whether to shuffle the learning table
        :param m: number of examples to return
        :param weigh: whether to return the sample weights [w]
        """
        n = self.N if data == "learning" else self.n_testing
        if n == 0:
            return None
        indices = np.arange(n)
        if shuff:
            np.random.shuffle(indices)
        indices = indices[:m]

        X, y = Frame.table(self, data)
        X = X[indices]
        y = self._embedding(y[indices])

        if weigh:
            return X, y, self.sample_weights[indices]
        return X, y

    def translate(self, preds, dummy=False):
        """
        Translates ANN predicions back to actual labels.

        :param preds: the ANN predictions (1D/2D array)
        :param dummy: if set, the dummycodes are returned
        """
        return self._embedding.translate(preds, dummy)

    def dummycode(self, data="testing", m=None):
        """
        Dummycodes the dependent (Y) variables

        :param data: which subset to dummycode
        :param m: number of examples returned
        """
        d = {"t": self.tindeps,
             "l": self.lindeps,
             "d": self.indeps}[data[0]]
        if m is None:
            m = d.shape[0]
        return self._embedding.dummycode(d[:m])

    @property
    def sample_weights(self):
        """
        Weigh samples according to category representedness in the data.
        Weights are centered around 1.0
        """
        rate_by_category = np.array([sum([cat == point for point in self.lindeps])
                                     for cat in self.categories]).astype(self.floatX)
        rate_by_category /= self.N
        rate_by_category = 1 - rate_by_category
        weight_dict = dict(zip(self.categories, rate_by_category))
        weights = np.vectorize(lambda cat: weight_dict[cat])(self.lindeps)
        weights -= weights.mean()
        weights += 1
        return weights

    @property
    def neurons_required(self):
        """
        Returns the required number of input and output neurons
        to process this dataset.
        """
        inshape, outshape = self.learning.shape[1:], self._embedding.outputs_required
        if isinstance(inshape, int):
            inshape = (inshape,)
        if isinstance(outshape, int):
            outshape = (outshape,)
        return inshape, outshape

    def average_replications(self):
        """
        This method is deprecated, but will be updated soon when multiple
        stored features become supported.
        """
        warnings.warn("Deprecated!", DeprecationWarning)
        replications = {}
        for i, indep in enumerate(self.indeps):
            if indep in replications:
                replications[indep].append(i)
            else:
                replications[indep] = [i]

        newindeps = np.fromiter(replications.keys(), dtype="<U4")
        newdata = {indep: np.mean(self.data[replicas], axis=0)
                   for indep, replicas in replications.items()}
        newdata = np.array([newdata[indep] for indep in newindeps])
        self.indeps = newindeps
        self.data = newdata
        self.reset_data(shuff=True, transform=True)

    def concatenate(self, other):
        """Concatenates two frames. Heavily asserts compatibility."""
        transformation, trparam = Frame.concatenate(self, other)
        if self.embedding != other.embedding:
            warnings.warn("The two data frames are embedded differently! Reverting to OneHot!")
            embedding = 0
        else:
            embedding = self._embedding.dim
        self.data = np.concatenate((self.data, other.data))
        self.indeps = np.concatenate((self.indeps, other.indeps))
        self.data.flags["WRITEABLE"] = False
        self.indeps.flags["WRITEABLE"] = False
        self.reset_data(shuff=False, embedding=embedding, transform=transformation, trparam=trparam)
