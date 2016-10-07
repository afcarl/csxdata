"""
Frames for holding multidimensional data and to interact with it.
Copyright (C) 2016  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import abc
import warnings

import numpy as np

from .utilities.const import floatX, roots, log
from .utilities.features import Transformation
from .utilities.nputils import shuffle
from .utilities.parsers import Parse


class _Data(abc.ABC):
    """
    Base class for Data Wrappers
    Can work with learning tables (X, y), plaintext files (.txt, .csv)
     and NumPy arrays.
    Also wraps whitening and other transformations (PCA, autoencoding, standardization).
    """

    def __init__(self, source, cross_val, indeps_n, header, sep, end):

        def parse_source():
            typeerrorstring = "Data wrapper doesn't support supplied data source!"
            if isinstance(source, np.ndarray):
                return Parse.array(source, header, indeps_n)
            elif isinstance(source, tuple):
                return Parse.learning_table(source)

            if not isinstance(source, str):
                raise TypeError(typeerrorstring)

            if "mnist.pkl.gz" == source.lower()[-12:]:
                from .utilities.parsers import mnist_tolearningtable
                return Parse.learning_table(mnist_tolearningtable(source))
            elif ".pkl.gz" in source.lower():
                return Parse.learning_table(source)
            elif source.lower()[-4:] in (".csv" or ".txt"):
                return Parse.csv(source, header, indeps_n, sep, end)
            else:
                raise TypeError(typeerrorstring)

        def determine_no_testing():
            # TODO: this might be a code duplication of the <crossval> property setter!
            err = ("Invalid value for cross_val! Can either be\nrate (float 0.0-1.0)\n" +
                   "number of testing examples (0 <= int <= len(data))\n" +
                   "the literals 'full', 'half' or 'quarter', or None.")
            if isinstance(cross_val, float):
                if not (0.0 <= cross_val <= 1.0):
                    raise ValueError(err)
                return int(self.data.shape[0] * cross_val)
            elif isinstance(cross_val, int):
                return cross_val
            elif isinstance(cross_val, str):
                cv = cross_val.lower()
                if cv not in ("full", "half", "quarter", "f", "h", "q"):
                    raise ValueError(err)
                return int(self.data.shape[0] * {"f": 1.0, "h": 0.5, "q": 0.25}[cv[0]])
            elif cross_val is None:
                return 0.0
            else:
                raise TypeError(err)

        self.learning = None
        self.testing = None
        self.lindeps = None
        self.tindeps = None
        self.type = None
        self._transformation = None
        self._transformed = False

        self._tmpdata = roots["cache"] + "tmpdata.pkl"
        self._tmpindeps = roots["cache"] + "tmpindeps.pkl"

        data, indeps, headers = parse_source()
        self.data = data
        self.indeps = indeps
        del data, indeps
        self.data.flags["WRITEABLE"] = False
        if self.indeps is not None:
            self.indeps.flags["WRITEABLE"] = False

        self.n_testing = determine_no_testing()
        self._crossval = cross_val
        self.headers = headers

    # TRIING TO MOVE THE CORE READ-ONLY DATA TO DISC AND TO IMPLEMENT DATA AS A PROPERTY
    # ----------------------------------------------------------------------------------
    # def _dump_data_to_cache(self, dat: np.ndarray=None, dep: np.ndarray=None):
    #     if (dat is None) == (dep is None):
    #         raise RuntimeError("Either supply data or indeps!")
    #     if dat is not None:
    #         np.save(self._tmpdata, arr=dat)
    #     if dep is not None:
    #         np.save(self._tmpindeps, arr=dep)
    #
    # @property
    # def data(self):
    #     X = np.load(self._tmpdata)
    #     return X
    #
    # @data.setter
    # def data(self, X):
    #     print("Setting data...")
    #     self._dump_data_to_cache(dat=X)
    #
    # @property
    # def indeps(self):
    #     Y = np.load(self._tmpindeps)
    #     return Y
    #
    # @indeps.setter
    # def indeps(self, Y):
    #     print("Setting indeps...")
    #     self._dump_data_to_cache(dep=Y)

    @property
    def transformation(self):
        out = self._transformation.name if self._transformation is not None else None
        return out

    @transformation.setter
    def transformation(self, transformation):
        # TODO: this might be a code duplication of <_Transformation.sanity_check()>
        er = ("Transformation not understood!\n" +
              "Please supply a tuple (name of transformation, parameters) " +
              "or the string literal 'std'")
        if isinstance(transformation, tuple):
            if len(transformation) == 1 and isinstance(transformation[0], str):
                transformation = transformation[0]

        if isinstance(transformation, str):
            tr = transformation[:5].lower()
            full = int(np.prod(self.learning.shape[1:]))
            if tr in ("std", "stand"):
                transformation = ("std", None)
            elif tr in ("pca", "princ"):
                transformation = ("pca", full)
            elif tr == "lda":
                transformation = ("lda", full)
            elif tr in ("ica", "indep"):
                transformation = ("ica", full)
            else:
                raise ValueError(er)

        if not isinstance(transformation, tuple):
            raise TypeError(er)

        if not len(transformation) == 2:
            raise ValueError(er)

        if not (isinstance(transformation[0], str) or transformation[0] is None):
            raise ValueError(er)

        if not (isinstance(transformation[1], int) or transformation[1] is None):
            raise ValueError(er)

        self.set_transformation(*transformation)

    def set_transformation(self, transformation: str, features):
        if self._transformed:
            warnings.warn("{} is already applied. Resetting previous transformation!")
            self.reset_data()
        if transformation[0] is None:
            self.reset_data(shuff=False, transform=transformation[0], trparam=None)

        self._transformation = {
            "std": Transformation.standardization,
            "stand": Transformation.standardization,
            "pca": Transformation.pca,
            "princ": Transformation.pca,
            "lda": Transformation.lda,
            "ica": Transformation.ica,
            "indep": Transformation.ica,
            "ae": Transformation.autoencoder,
            "autoe": Transformation.autoencoder
        }[transformation[:5].lower()](features)

        if transformation == "lda":
            self._transformation.fit(self.learning, self.lindeps)
        else:
            self._transformation.fit(self.learning)

        self.learning = self._transformation(self.learning)
        if self.n_testing > 0:
            self.testing = self._transformation(self.testing)
        self._transformed = True

    def transform(self, X: np.ndarray):
        return self._transformation(X)

    def table(self, data, m=None):
        """Returns a learning table (X, y)

        :param data: which partition of data to use (learning, testing, or the untransformed original data)
        :param m: the number of data points to return

        :returns: X, y NumPy NDarrays as the dependent and independent variables
        """

        data = data[0].lower()
        if data not in ("l", "t"):
            raise RuntimeError("Unkown data subset! Choose either <learning> or <testing>!")

        if data == "t" and self.n_testing == 0:
            warnings.warn("There is no testing data!")
            return

        X = self.learning if data == "l" else self.testing
        y = self.lindeps if data == "l" else self.tindeps

        return X[:m], y[:m]

    def batchgen(self, bsize: int, data: str="learning", infinite=False) -> np.ndarray:
        """Returns a generator that yields batches randomly from the
        specified dataset.

        :param bsize: specifies the size of the batches
        :param data: specifies the data partition (learning, testing or data)
        :param infinite: if set to True, the generator becomes infinite.
        """
        tab = shuffle(self.table(data))
        if tab is None:
            return

        tsize = len(tab[0])
        start = 0
        end = start + bsize

        while True:
            if end > tsize:
                end = tsize
            if start < tsize:
                if infinite:
                    start = 0
                    end = start + bsize
                else:
                    break

            out = (tab[0][start:end], tab[1][start:end])

            start += bsize
            end += bsize

            yield out

    @abc.abstractmethod
    def reset_data(self, shuff: bool, transform, trparam: int=None):
        """Resets any transformations and partitioning previously applied.

        :param shuff: whether the partitioned data should be shuffled or not
        :param transform: what transformation to apply
        :param trparam: arguments if transform needs them
        """

        if shuffle:
            dat, ind = shuffle((self.data, self.indeps))
        else:
            dat, ind = self.data, self.indeps

        if self.n_testing > 0:
            self.learning = dat[self.n_testing:]
            self.testing = dat[:self.n_testing]
            self.lindeps = ind[self.n_testing:] if ind is not None else None
            self.tindeps = ind[:self.n_testing] if ind is not None else None
        else:
            self.learning = dat
            self.lindeps = ind if ind is not None else None
            self.testing = None
            self.tindeps = None

        if transform is True:
            if self._transformation is None:
                return
            if trparam is None:
                self.set_transformation(self._transformation.name, self._transformation.params)
            else:
                self.set_transformation(self._transformation.name, trparam)
        elif isinstance(transform, str):
            self.set_transformation(transform, trparam)
        elif not transform or transform == "None":
            self._transformed = False
            self._transformation = None
        else:
            raise RuntimeError("Specified transformation was not understood!")

    @abc.abstractproperty
    def neurons_required(self):
        return None

    @property
    def N(self):
        return self.learning.shape[0]

    @property
    def crossval(self):
        return self._crossval

    @crossval.setter
    def crossval(self, alpha):
        if alpha == 0:
            self._crossval = 0.0
        elif isinstance(alpha, int) and alpha == 1:
            print("Received an integer value of 1. Assuming 1 testing sample!")
            self._crossval = 1 / self.data.shape[0]
        elif isinstance(alpha, int) and alpha > 1:
            self._crossval = alpha / self.data.shape[0]
        elif isinstance(alpha, float) and 0.0 < alpha <= 1.0:
            self._crossval = alpha
        else:
            raise ValueError("Wrong value supplied! Give the ratio (0.0 <= alpha <= 1.0)\n" +
                             "or the number of samples to be used for validation!")
        self.n_testing = int(self.data.shape[0] * self._crossval)
        self.reset_data(shuff=True, transform=True)

    @crossval.deleter
    def crossval(self):
        self._crossval = 0.0
        self.n_testing = 0

    @abc.abstractmethod
    def concatenate(self, other):
        dimerror = "Dimensions are different! Can't concatenate..."
        dtypeerror = "Data types are different! Can't concatenate..."

        if not self.data.ndim == other.data.ndim:
            raise TypeError(dimerror)
        if any([dim1 != dim2 for dim1, dim2 in zip(self.data.shape[1:], other.data.shape[1:])]):
            raise TypeError(dimerror)
        if any([dim1 != dim2 for dim1, dim2 in zip(self.indeps.shape[1:], other.indeps.shape[1:])]):
            raise TypeError(dimerror)
        if self.data.dtype != other.data.dtype:
            raise TypeError(dtypeerror)
        if self.indeps.dtype != other.indeps.dtype:
            raise TypeError(dtypeerror)
        if self.transformation != other.transformation:
            warnings.warn("Supplied data frames are transformed differently. Reverting transformation!")
        transformation = self.transformation
        if transformation:
            trparam = self._transformation.params
        else:
            trparam = None
        return transformation, trparam


class CData(_Data):
    """
    This class is for holding categorical learning myData.
    """

    def __init__(self, source, cross_val=.2, header=True, sep="\t", end="\n",
                 standardize=False, pca=0, autoencode=0, embedding=None):

        def sanitize_independent_variables():
            # In categorical data, there is only 1 independent categorical variable
            # which is stored in a 1-tuple or 1 element vector. We free it from its misery
            if type(self.indeps[0]) in (np.ndarray, tuple, list):
                self.indeps = np.array([d[0] for d in self.indeps])

        def parse_transformation():
            if autoencode:
                return "ae", autoencode
            elif pca:
                return "pca", pca
            elif standardize:
                return "std", None
            else:
                return False, None

        def get_categories():
            if isinstance(self.indeps, np.ndarray):
                idps = self.indeps.tolist()
            elif isinstance(self.indeps, list) or isinstance(self.indeps, tuple):
                idps = self.indeps
            else:
                raise RuntimeError("Cannot parse categories!")
            return list(set(idps))

        _Data.__init__(self, source, cross_val, 1, header, sep, end)

        sanitize_independent_variables()

        self.type = "classification"
        self.categories = get_categories()
        self._embedding = None

        tr, trparam = parse_transformation()

        self.reset_data(shuff=False, embedding=embedding, transform=tr, trparam=trparam)

    @property
    def embedding(self):
        return self._embedding.name

    @embedding.setter
    def embedding(self, emb):
        from .utilities.features import Embed, OneHot

        if emb in (None, "None"):
            emb = 0
        elif isinstance(emb, str):
            if emb.lower() == "onehot":
                emb = 0
            else:
                raise RuntimeError("Embedding not understood!")
        if not isinstance(emb, int):
            raise RuntimeError("Embedding not understood!")

        if emb:
            self._embedding = Embed(embeddim=emb)
        else:
            self._embedding = OneHot()

        self._embedding.fit(self.indeps)

    @embedding.deleter
    def embedding(self):
        self.embedding = 0

    def reset_data(self, shuff: bool=True, embedding=0, transform=None, trparam: int=None):
        _Data.reset_data(self, shuff, transform, trparam)

        self.embedding = embedding

    def batchgen(self, bsize: int, data: str="learning", weigh=False, infinite=False):
        tab = self.table(data, weigh=weigh)
        n = len(tab[0])
        start = 0
        end = start + bsize

        def slice_elements(lt, begin, stop):
            return tuple(map(lambda elem: elem[begin:stop], lt))

        while 1:
            if end >= n:
                end = n
            if start >= n:
                if infinite:
                    start = 0
                    end = start + bsize
                else:
                    break

            # This is X y (w) with dim[0] = bsize
            out = slice_elements(tab, start, end)

            start += bsize
            end += bsize

            yield out

    def table(self, data="learning", shuff=True, m=None, weigh=False):
        """Returns a learning table"""
        n = self.N if data == "learning" else self.n_testing
        if n == 0:
            return None
        indices = np.arange(n)
        if shuff:
            np.random.shuffle(indices)
        indices = indices[:m]

        X, y = _Data.table(self, data)
        X = X[indices]
        y = self._embedding(y[indices])

        if weigh:
            return X, y, self.sample_weights[indices]
        return X, y

    def translate(self, preds: np.ndarray, dummy=False):
        """Translates a Brain's predictions to a human-readable answer"""
        return self._embedding.translate(preds, dummy)

    def dummycode(self, data="testing", m=None):
        d = {"t": self.tindeps,
             "l": self.lindeps,
             "d": self.indeps}[data[0]]
        if m is None:
            m = d.shape[0]
        return self._embedding.dummycode(d[:m])

    @property
    def sample_weights(self):
        rate_by_category = np.array([sum([cat == point for point in self.lindeps])
                                     for cat in self.categories]).astype(floatX)
        rate_by_category /= self.N
        # assert np.sum(rate_by_category) == 1.0, "Category weight determination failed!"
        rate_by_category = 1 - rate_by_category
        weight_dict = dict(zip(self.categories, rate_by_category))
        weights = np.vectorize(lambda cat: weight_dict[cat])(self.lindeps)
        weights -= weights.mean()
        weights += 1
        return weights

    @property
    def neurons_required(self):
        """Returns the required number of input and output neurons
         to process this dataset"""
        inshape, outshape = self.learning.shape[1:], self._embedding.outputs_required
        if len(inshape) == 1:
            inshape = inshape[0]
        return inshape, outshape

    def average_replications(self):
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
        transformation, trparam = _Data.concatenate(self, other)
        if self.embedding != other.embedding:
            warnings.warn("The two data frames are embedded differently! Reverting!")
            embedding = 0
        else:
            embedding = self._embedding.dim
        self.data = np.concatenate((self.data, other.data))
        self.indeps = np.concatenate((self.indeps, other.indeps))
        self.data.flags["WRITEABLE"] = False
        self.indeps.flags["WRITEABLE"] = False
        self.reset_data(shuff=False, embedding=embedding, transform=transformation, trparam=trparam)


class RData(_Data):
    """
    Class for holding regression type data.
    """

    def __init__(self, source, cross_val, indeps_n, header, sep=";", end="\n",
                 standardize=False, autoencode=0, pca=0):
        _Data.__init__(self, source, cross_val, indeps_n, header, sep, end)

        self.type = "regression"
        self._downscaled = False

        self._oldfctrs = None
        self._newfctrs = None

        self.indeps = np.atleast_2d(self.indeps)

        if autoencode:
            self.set_transformation("ae", autoencode)
        elif pca:
            self.set_transformation("pca", pca)
        elif standardize:
            self.set_transformation("std", 0)

        self.reset_data(shuff=False, transform=False, trparam=None)

    def reset_data(self, shuff=True, transform=None, trparam=None):
        _Data.reset_data(self, shuff, transform, trparam)
        if not self._downscaled:
            from .utilities.nputils import featscale

            self.lindeps, self._oldfctrs, self._newfctrs = \
                featscale(self.lindeps, axis=0, ufctr=(0.1, 0.9), return_factors=True)
            self._downscaled = True
            self.tindeps = self.downscale(self.tindeps)
        self.indeps = self.indeps.astype(floatX)
        self.lindeps = self.lindeps.astype(floatX)
        self.tindeps = self.tindeps.astype(floatX)

    def _scale(self, A, where):
        def sanitize():
            assert self._downscaled, "Scaling factors not yet set!"
            assert where in ("up", "down"), "Something is very weird here..."
            if where == "up":
                return self._newfctrs, self._oldfctrs
            else:
                return self._oldfctrs, self._newfctrs

        from .utilities.nputils import featscale
        fctr_list = sanitize()
        return featscale(A, axis=0, dfctr=fctr_list[0], ufctr=fctr_list[1])

    def upscale(self, A):
        return self._scale(A, "up")

    def downscale(self, A):
        return self._scale(A, "down")

    @property
    def neurons_required(self):
        fanin, outshape = self.learning.shape[1:], self.lindeps.shape[1]
        if len(fanin) == 1:
            fanin = fanin[0]
        return fanin, outshape

    def concatenate(self, other):
        transform, trparam = _Data.concatenate(self, other)
        self.data = np.concatenate((self.data, other.data))
        self.indeps = np.concatenate((self.indeps, other.indeps))
        self._downscaled = False
        self.reset_data(shuff=False, transform=transform, trparam=trparam)


class Sequence(_Data):
    def __init__(self, source, embeddim=None, cross_val=0.2, n_gram=1, timestep=None, coding="utf-8-sig"):
        from .utilities.features import Embedding

        def set_embedding(d):
            if embeddim:
                self._embedding = Embedding.embed(embeddim)
            else:
                self._embedding = Embedding.onehot()
            self._embedding.fit(d)
            return self._embedding(d)

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
                d = [[e for e in time[:-1]] for time in dat]
                dp = [time[-1] for time in dat]
            return d, dp

        self._embedding = None
        self.timestep = timestep
        data = Parse.txt(source, ngram=n_gram, coding=coding)
        data = set_embedding(data)
        data, deps = split_X_y(data)

        _Data.__init__(self, (data, deps), cross_val=cross_val, indeps_n=0, header=None, sep=None, end=None)
        self.reset_data(shuff=True)

    def reset_data(self, shuff: bool, transform=None, trparam: int=None):
        if transform is not None:
            transform = None
        _Data.reset_data(self, shuff=shuff, transform=transform)

    @property
    def neurons_required(self):
        return (self.timestep - 1, self._embedding.dim), self._embedding.dim

    def translate(self, preds, use_proba=False):
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

        human = self._embedding.translate(preds)
        return human

    def primer(self):
        from random import randrange
        primer = self.learning[randrange(self.N)]
        return primer.reshape(1, *primer.shape)

    def concatenate(self, other):
        pass


class MassiveSequence:
    def __init__(self, source, embeddim=None, cross_val=0.2, n_gram=1, timestep=None, coding="utf-8-sig"):
        from .utilities.features import Embedding

        def set_embedding():
            if embeddim:
                self._embedding = Embedding.embed(embeddim)
            else:
                self._embedding = Embedding.onehot()

        def chop_up_to_timesteps():
            newN = self.data.shape[0] // timestep
            if self.data.shape[0] % timestep != 0:
                warnings.warn("Trimming data to fit into timesteps!", RuntimeWarning)
                self.data = self.data[:self.data.shape[0] - (self.data.shape[0] % timestep)]
            newshape = newN, timestep
            print("Reshaping from: {} to: {}".format(self.data.shape, newshape))
            self.data = self.data.reshape(*newshape)

        self._embedding = None
        self.timestep = timestep
        self._crossval = cross_val

        self.data = np.ravel(Parse.txt(source, ngram=n_gram, coding=coding))
        set_embedding()
        self._embedding.fit(self.data)
        chop_up_to_timesteps()

        self.n_testing = int(self.data.shape[0] * cross_val)
        self.N = self.data.shape[0]

    @property
    def neurons_required(self):
        return (self.timestep - 1, self._embedding.dim), self._embedding.dim

    def translate(self, preds, use_proba=False):
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

        human = self._embedding.translate(preds)
        return human

    def batchgen(self, bsize=None):
        MIL = 10000
        if bsize is None:
            bsize = MIL
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
