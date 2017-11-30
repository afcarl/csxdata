"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""
import warnings

import numpy as np
from .vectorop import ravel_to_matrix as rtm, dummycode


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


def transform(X, factors, get_model, method: str, y=None):
    if method == "raw" or method is None:
        return X
    if not factors or factors == "full" or not isinstance(factors, int):
        factors = np.prod(X.shape[1:])
        if method == "lda":
            factors -= 1

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

    latent = model.fit_transform(rtm(X), y)

    if isinstance(latent, tuple):
        latent = latent[0]
    return (latent, model) if get_model else latent


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


def tf_haversine():
    """Returns a reference to the compiled Haversine distance function"""
    import tensorflow as tf

    from .vectorop import floatX

    coords1 = tf.placeholder(dtype=floatX, shape=(None, 2), name="Coords1")
    coords2 = tf.placeholder(dtype=floatX, shape=(None, 2), name="Coords2")

    R = np.array([6367], dtype="int32")  # Approximate radius of Mother Earth in kms
    coords1 = np.deg2rad(coords1)
    coords2 = np.deg2rad(coords2)
    lon1, lat1 = coords1[:, 0], coords1[:, 1]
    lon2, lat2 = coords2[:, 0], coords2[:, 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = tf.sin(dlat / 2) ** 2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2) ** 2
    e = 2 * tf.asin(tf.sqrt(d))
    d_haversine = e * R
    return d_haversine


