from .transformation import *
from .embedding import *


class Transformation:
    @staticmethod
    def pca(factors):
        return PCA(factors)

    @staticmethod
    def autoencoder(features):
        return Autoencoding(features)

    @staticmethod
    def standardization(features=None):
        del features
        return Standardization()

    @staticmethod
    def lda(features=None):
        return LDA(features)

    @staticmethod
    def ica(features=None):
        return ICA(factors=features)

    def __getitem__(self, item):
        if item not in self.__dict__:
            raise IndexError("Unknown transformation:", item)
        return self.__dict__[item]


class Embedding:
    @classmethod
    def onehot(cls, yes=None, no=None):
        return OneHot(yes, no)

    @classmethod
    def embed(cls, embeddim):
        return Embed(embeddim)
