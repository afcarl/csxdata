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

from .const import floatX
from .utilities.nputils import argshuffle, shuffle
from .utilities.features import Transformation


class _Data(abc.ABC):
    """
    Base class for Data Wrappers
    Can work with learning tables (X, y), plaintext files (.txt, .csv)
     and NumPy arrays.
    Also wraps whitening and other transformations (PCA, autoencoding, standardization).
    """

    def __init__(self, source, cross_val, indeps_n, header, sep, end):

        def parse_source():
            from .utilities.parsers import Parse
            if isinstance(source, np.ndarray):
                return Parse.array(source, header, indeps_n)
            if isinstance(source, tuple):
                return Parse.learning_table(source)
            elif "lt.pkl.gz" in source.lower():
                return Parse.learning_table(source)
            elif "mnist.pkl.gz" == source.lower()[-12:]:
                from .utilities.parsers import mnist_tolearningtable
                return Parse.learning_table(mnist_tolearningtable(source))
            elif source.lower()[-4:] in (".csv" or ".txt"):
                return Parse.csv(source, header, indeps_n, sep, end)
            else:
                raise TypeError("Data wrapper doesn't support supplied data source!")

        def determine_no_testing():
            # TODO: this might be a code duplication of the <crossval> property setter!
            err = ("Invalid value for cross_val! Can either be\nrate (float 0.0-1.0)\n" +
                   "number of testing examples (0 <= int <= len(data))\n" +
                   "the literals 'full', 'half' or 'quarter', or None.")
            if isinstance(cross_val, float):
                if not (0.0 <= cross_val <= 1.0):
                    raise ValueError(err)
                return int(data.shape[0] * cross_val)
            elif isinstance(cross_val, int):
                return cross_val
            elif isinstance(cross_val, str):
                cv = cross_val.lower()
                if cv not in ("full", "half", "quarter", "f", "h", "q"):
                    raise ValueError(err)
                return int(data.shape[0] * {"f": 1.0, "h": 0.5, "q": 0.25}[cv[0]])
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

        data, indeps, headers = parse_source()
        self.n_testing = determine_no_testing()
        self._crossval = cross_val

        self.headers = headers
        # TODO: data and indeps should be properties attached to a generator
        self.data, self.indeps = np.copy(data), np.copy(indeps)
        self.data.flags["WRITEABLE"] = False
        self.indeps.flags["WRITEABLE"] = False

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
            if transformation[:5].lower() in ("std", "stand"):
                transformation = ("std", None)
            elif transformation[:5].lower() in ("pca", "princ"):
                transformation = ("pca", int(np.prod(self.learning.shape[1:])))
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
        if transformation[0] in (None, "None"):
            self.reset_data(shuff=False, transform=None, param=None)

        self._transformation = {
            "std": Transformation.standardization,
            "stand": Transformation.standardization,
            "pca": Transformation.pca,
            "princ": Transformation.pca,
            "ae": Transformation.autoencoder,
            "autoe": Transformation.autoencoder
        }[transformation[:5].lower()](self, features)

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
        X = {"l": self.learning,
             "t": self.testing,
             "d": self.data}[data[0]]
        y = {"l": self.lindeps,
             "t": self.tindeps,
             "d": self.indeps}[data[0]]

        if m is not None:
            X, y = X[:m], y[:m]

        return X, y

    def batchgen(self, bsize: int, data: str="learning") -> np.ndarray:
        """Returns a generator that yields batches randomly from the
        specified dataset.

        :param bsize: specifies the size of the batches
        :param data: specifies the data partition (learning, testing or data)
        """
        tab = shuffle(self.table(data))
        tsize = len(tab[0])
        start = 0
        end = start + bsize

        while start < tsize:
            if end > tsize:
                end = tsize

            out = (tab[0][start:end], tab[1][start:end])

            start += bsize
            end += bsize

            yield out

    @abc.abstractmethod
    def reset_data(self, shuff: bool, transform, param: int=None):
        """Resets any transformations and partitioning previously applied.

        :param shuff: whether the partitioned data should be shuffled or not
        :param transform: what transformation to apply
        :param param: arguments if transformation needs them
        """

        if shuffle:
            dat, ind = shuffle((self.data, self.indeps))
        else:
            dat, ind = self.data, self.indeps

        if self.n_testing > 0:
            self.learning = dat[self.n_testing:]
            self.lindeps = ind[self.n_testing:]
            self.testing = dat[:self.n_testing]
            self.tindeps = ind[:self.n_testing]
        else:
            self.learning = dat
            self.lindeps = ind
            self.testing = None
            self.tindeps = None

        if transform is True:
            if self._transformation is None:
                return
            if param is None:
                self.set_transformation(self._transformation.name, self._transformation.params)
            else:
                self.set_transformation(self._transformation.name, param)
        elif isinstance(transform, str):
            self.set_transformation(transform, param)
        elif transform in (None, False):
            self._transformed = False
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
            self._embedding = Embed(master=self, embeddim=emb)
        else:
            self._embedding = OneHot(master=self)

    @embedding.deleter
    def embedding(self):
        self.embedding = 0

    def reset_data(self, shuff: bool=True, embedding=0, transform=None, trparam: int=None):
        _Data.reset_data(self, shuff, transform, trparam)

        self.embedding = embedding

    def batchgen(self, bsize: int, data: str="learning", weigh=False):
        tab = self.table(data, weigh=weigh)
        m = len(tab[0])
        start = 0
        end = start + bsize

        def slice_elements(lt, begin, stop):
            return tuple(map(lambda elem: elem[begin:stop], lt))

        while start < m:
            if end > m:
                end = m
            out = slice_elements(tab, start, end)

            start += bsize
            end += bsize

            yield out

    def table(self, data="learning", shuff=True, m=None, weigh=False):
        """Returns a learning table"""
        lt = _Data.table(self, data, m)
        if shuff:
            indices = argshuffle(lt)
        else:
            indices = np.arange(lt[0].shape[0])
        X, indep = self.learning[indices], self.lindeps[indices]
        y = self._embedding(indep)
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
        assert np.sum(rate_by_category) == 1.0, "Category weight determination failed!"
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
        return self.learning.shape[1:], self._embedding.outputs_required

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


class Sequence:
    def __init__(self, source):
        self._raw = source
        self._vocabulary = dict()
        self.data = None
        self.embedded = False
        self.tokenized = False
        self.N = len(self._raw)

    def embed(self, dims):
        assert not (self.tokenized or self.embedded)
        self._encode("embed", dims)
        self.embedded = True

    def tokenize(self):
        assert not (self.tokenized or self.embedded)
        self._encode("tokenize")
        self.tokenized = True

    def _encode(self, how, dims=0):
        symbols = list(set(self._raw))
        if how == "tokenize":
            embedding = np.eye(len(symbols), len(symbols))
        elif how == "embed":
            assert dims, "Dims unspecified!"
            embedding = np.random.random((len(symbols), dims)).astype(floatX)
        else:
            raise RuntimeError("Something is not right!")
        self._vocabulary = dict(zip(symbols, embedding))
        self.data = np.array([self._vocabulary[x] for x in self._raw])

    def table(self):
        return ([word[1:] for word in self._raw],
                [word[:-1] for word in self._raw])

    def batchgen(self, size):
        assert self.embedded ^ self.tokenized
        for step in range(self.data.shape[0] // size):
            start = step * size
            end = start + size
            if end > self.data.shape[0]:
                end = self.data.shape[0]
            if start >= self.data.shape[0]:
                break

            sentence = self.data[start:end]

            yield sentence[:-1], sentence[1:]

    def neurons_required(self):
        return self.data.shape[-1], self.data.shape[-1]


class Text(Sequence):
    def __init__(self, source, limit=8000, vocabulary=None,
                 embed=False, embeddim=0, tokenize=False):
        Sequence.__init__(self, source)
        self._raw = source
        self._tokenized = False
        self._embedded = False
        self._util_tokens = {"unk": "<UNK>",
                             "start": "<START>",
                             "end": "<END>"}
        self._dictionary = dict()
        self._vocabulary = vocabulary if vocabulary else {}

        self.data = None

        if embed and tokenize:
            raise RuntimeError("Please choose either embedding or tokenization, not both!")

        if embed or tokenize:
            if embed and not embeddim:
                warnings.warn("Warning! Embedding vector dimension unspecified, assuming 5!",
                              RuntimeWarning)
                embeddim = 5
            self.initialize(vocabulary, limit, tokenize, embed, embeddim)

    def neurons_required(self):
        return self.data.shape[0]  # TODO: is this right??

    def initialize(self, vocabulary: dict=None, vlimit: int=None,
                   tokenize: bool=True, embed: bool=False,
                   embeddim: int=5):

        def accept_vocabulary():
            example = list(vocabulary.keys())[0]
            dtype = example.dtype
            if "float" in dtype:
                self._embedded = True
                self._tokenized = False
            elif "int" in dtype:
                self._embedded = False
                self._tokenized = True
            else:
                raise RuntimeError("Wrong vocabulary format!")
            self._vocabulary = vocabulary
            self.data = build_learning_data()

        def ask_for_input():
            v = None
            while 1:
                v = input("Please select one:\n(E)mbed\n(T)okenize\n> ")
                if v[0].lower() in "et":
                    break
            return v == "t", v == "e"

        def build_vocabulary():
            words = dict()
            for sentence in self._raw:
                for word in sentence:
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1
            words = [word for word in sorted(list(words.items()), key=lambda x: x[1], reverse=True)][:vlimit]

            if tokenize:
                emb = np.eye(len(words), len(words))
            else:
                emb = np.random.random((len(words), embeddim))

            voc = {word: array for word, array in zip(words, emb)}
            dic = {i: word for i, word in enumerate(words)}

            return voc, dic

        def build_learning_data():
            from .const import UNKNOWN
            data = []
            for i, sentence in enumerate(self._raw):
                s = []
                for word in sentence:
                    if word in self._vocabulary:
                        s.append(self._vocabulary[word])
                    else:
                        s.append(UNKNOWN)
                data.append(np.array(s))
            return data

        if vocabulary:
            accept_vocabulary()
            return

        if not (tokenize or embed) or (tokenize and embed):
            tokenize, embed = ask_for_input()

        self._vocabulary, self._dictionary = build_vocabulary()

        self.data = build_learning_data()

    def batchgen(self, size=None):
        sentences = np.copy(self.data)
        np.random.shuffle(sentences)
        for sentence in sentences:
            yield sentence[:-1], sentence[1:]


class Text2:
    def __init__(self, raw: str, dictionary: dict, embed=0):
        self._raw = raw
        self._dictionary = dictionary
        self._emb = embed
        self._keys = None
        self._values = None

        self.data = [self._dictionary[word] for word in self._raw]

    @classmethod
    def characterwise(cls, chain: str, vocabulary=None, embed=0):
        if vocabulary is None:
            vocabulary = ["<WORD_START>"] + list(set(chain)) + ["<WORD_END>"]
        if embed:
            print("Embedding characters into {} dimensional space!".format(embed))
            embedding = sorted(np.random.randn(len(vocabulary), embed).astype(floatX),
                               key=lambda x: x.sum())
        else:
            print("Tokenizing characters...")
            embedding = np.eye(len(vocabulary)).astype(floatX)
        embedding = dict(zip(sorted(vocabulary), embedding))
        return Text2(chain, embedding)

    @property
    def neurons_required(self):
        embeddim = self._dictionary[0].shape[1]
        return embeddim, embeddim

    @property
    def encoding(self):
        embdim = self._dictionary[0].shape[1]
        return "Embedding ({})".format(embdim) if self._emb else "Tokenization ({})".format(embdim)

    def table(self):
        return [ch for ch in self.data[:-1]], [ch for ch in self.data[1:]]

    def translate(self, output):
        def from_tokenization():
            if self._keys is None:
                keys, values = list(zip(*sorted(
                    [(k, np.argmax(v)) for k, v in self._dictionary.items], key=lambda x: x[1]
                )))
                self._keys = keys
            return self._keys[np.argmax(output)]

        def from_embedding():
            from .utilities.nputils import euclidean

            if self._keys is None or self._values is None:
                self._keys, self._values = list(zip(*list(self._dictionary.items())))
            return self._keys[np.argmin(
                [euclidean([output for _ in range(len(self._values))], self._values)]
            )]

        output = from_embedding() if self.encoding.lower()[0] == "e" else from_tokenization()
        return output
