"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""
import warnings

import numpy as np
from .vectorop import ravel_to_matrix as rtm, dummycode


_axlabels = {"pca": ("PC1", "PC2", "PC3"),
             "lda": ("LD1", "LD2", "LD3"),
             "ica": ("IC1", "IC2", "IC3")}


def autoencode(X: np.ndarray, hiddens=60, validation=None, epochs=30, get_model=False):

    from brainforge import BackpropNetwork, LayerStack
    from brainforge.layers import DenseLayer

    from .vectorop import standardize

    def sanitize(ftrs):
        if isinstance(hiddens, int):
            ftrs = (hiddens,)
        return ftrs

    def build_encoder(hid):
        dims = data.shape[1]
        encstack = LayerStack(dims, layers=[
            DenseLayer(hid[0], activation="tanh")
        ])
        if len(hid) > 1:
            for neurons in hid[1:]:
                encstack.add(DenseLayer(neurons, activation="tanh"))
            for neurons in hid[-2:0:-1]:
                encstack.add(DenseLayer(neurons, activation="tanh"))
        encstack.add(DenseLayer(dims, activation="linear"))
        return BackpropNetwork(encstack, cost="mse", optimizer="momentum")

    def std(training_data, test_data):
        training_data, (average, st_deviation) = standardize(rtm(training_data), return_factors=True)
        if test_data is not None:
            test_data = standardize(rtm(test_data), mean=average, std=st_deviation)
            test_data = (test_data, test_data)
        return training_data, test_data, (average, st_deviation)

    print("Creating autoencoder model...")

    hiddens = sanitize(hiddens)
    data, validation, transf = std(X, validation)

    autoencoder = build_encoder(hiddens)

    print("Initial loss: {}".format(autoencoder.evaluate(data, data)))

    autoencoder.fit(data, data, batch_size=20, epochs=epochs, validation=validation)
    model = autoencoder.get_weights(unfold=False)
    encoder, decoder = model[:len(hiddens)], model[len(hiddens):]

    transformed = np.tanh(data.dot(encoder[0][0]) + encoder[0][1])
    if len(encoder) > 1:
        for weights, biases in encoder[1:]:
            transformed = np.tanh(transformed.dot(weights) + biases)
    if get_model:
        return transformed, (encoder, decoder), transf
    else:
        return transformed


def transform(X, factors, get_model, method, y=None):
    if method == "raw" or method is None:
        return X
    if not factors or factors == "full":
        factors = np.prod(X.shape[1:])
        if method == "lda":
            factors -= 1

    if not isinstance(method, str):
        raise RuntimeError("Please supply a method name (pca, lda, ica, cca, pls)")
    method = method.lower()

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=factors, whiten=True)
    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        model = LDA(n_components=factors)
    elif method == "ica":
        from sklearn.decomposition import FastICA as ICA
        model = ICA(n_components=factors)
    elif method == "cca":
        from sklearn.cross_decomposition import CCA
        model = CCA(n_components=factors)
    elif method == "pls":
        from sklearn.cross_decomposition import PLSRegression as PLS
        model = PLS(n_components=factors)
        if str(y.dtype)[:3] not in ("flo", "int"):
            y = dummycode(y, get_translator=False)
    else:
        raise ValueError("Method {} unrecognized!".format(method))

    X = rtm(X)
    if method in ("lda", "cca", "pls"):
        if y is None:
            raise RuntimeError("y must be supplied for {}!".format(method))
        latent = model.fit_transform(X, y)
    else:
        if y is not None:
            warnings.warn("y supplied for {}. Ignoring!".format(method))
        latent = model.fit_transform(X)

    if isinstance(latent, tuple):
        latent = latent[0]
    if get_model:
        return latent, model
    else:
        return latent


def image_to_array(imagepath):
    """Opens an image file and returns it as a NumPy array of pixel values"""
    from PIL import Image
    return np.array(Image.open(imagepath))


def image_sequence_to_array(imageroot, outpath=None, generator=False):
    """Opens and merges an image sequence into a 3D tensor"""
    import os

    flz = os.listdir(imageroot)

    print("Merging {} images to 3D array...".format(len(flz)))
    if not generator:
        ar = np.stack([image_to_array(imageroot + image) for image in sorted(flz)])
        if outpath is not None:
            try:
                ar.dump(outpath)
            except MemoryError:
                warnings.warn("OOM, skipped array dump!", ResourceWarning)
            else:
                print("Images merged and dumped to {}".format(outpath))
        return ar
    for image in sorted(flz):
        yield image_to_array(imageroot + image)


def th_haversine():
    """Returns a reference to the compiled Haversine distance function"""
    from theano import tensor as T
    from theano import function

    from .vectorop import floatX

    coords1 = T.matrix("Coords1", dtype=floatX)
    coords2 = T.matrix("Coords2", dtype=floatX)

    R = np.array([6367], dtype="int32")  # Approximate radius of Mother Earth in kms
    coords1 = T.deg2rad(coords1)
    coords2 = T.deg2rad(coords2)
    lon1, lat1 = coords1[:, 0], coords1[:, 1]
    lon2, lat2 = coords2[:, 0], coords2[:, 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = T.sin(dlat / 2) ** 2 + T.cos(lat1) * T.cos(lat2) * T.sin(dlon / 2) ** 2
    e = 2 * T.arcsin(T.sqrt(d))
    d_haversine = e * R
    f_ = function([coords1, coords2], outputs=d_haversine)
    return f_


def projection(method, factors, X, Y, ellipse_sigma=0, **kw):
    trX = transform(X, factors, False, method, Y)
    plot(trX, Y, ellipse_sigma=ellipse_sigma,
         axlabels=_axlabels.get(method.lower(),
                                ("Factor01", "Factor02", "Factor03")),
         **kw)
