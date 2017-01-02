"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""
import warnings

import numpy as np
from .vectorops import ravel_to_matrix as rtm, dummycode


def autoencode(X: np.ndarray, hiddens, validation: np.ndarray=None, epochs=5,
               get_model: bool=False):
    """Autoencodes X with a dense autoencoder, built with the Keras ANN Framework"""

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import RMSprop

    from .vectorops import standardize

    def sanitize(ftrs):
        if isinstance(hiddens, int):
            ftrs = (hiddens,)
        return ftrs

    def build_encoder(hid):
        dims = data.shape[1]
        enc = Sequential()
        enc.add(Dense(input_dim=dims, output_dim=hid[0],
                      activation="tanh"))
        if len(hid) > 1:
            for neurons in hid[1:]:
                enc.add(Dense(output_dim=neurons, activation="tanh"))
            for neurons in hid[-2:0:-1]:
                enc.add(Dense(output_dim=neurons, activation="tanh"))
        enc.add(Dense(output_dim=dims, activation="tanh"))
        enc.compile(RMSprop(), loss="mse")
        # enc.compile(Adagrad(), loss="mse")
        return enc

    def std(training_data, test_data):
        training_data, (average, st_deviation) = standardize(rtm(training_data), return_factors=True)
        if test_data is not None:
            test_data = standardize(rtm(test_data), mean=average, std=st_deviation)
            test_data = (test_data, test_data)
        return training_data, test_data, (average, st_deviation)

    print("Creating autoencoder model...")

    hiddens = sanitize(hiddens)
    data, validation, transf = std(X, validation)

    encoder = build_encoder(hiddens)

    print("Initial loss: {}".format(encoder.evaluate(data, data, verbose=0)))

    encoder.fit(data, data, batch_size=20, nb_epoch=epochs, validation_data=validation)
    model = [layer.get_weights() for layer in encoder.layers]
    (encoder, decoder) = model[:len(hiddens)], model[len(hiddens):]

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


def image_sequence_to_array(imageroot, outpath=None):
    """Opens and merges an image sequence into a 3D tensor"""
    import os

    flz = os.listdir(imageroot)

    print("Merging {} images to 3D array...".format(len(flz)))
    ar = np.stack([image_to_array(imageroot + image) for image in sorted(flz)])

    if outpath is not None:
        ar.dump(outpath)
        print("Images merged and dumped to {}".format(outpath))

    return ar


def th_haversine():
    """Returns a reference to the compiled Haversine distance function"""
    from theano import tensor as T
    from theano import function

    from .vectorops import floatX

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


def plot(points, dependents, axlabels, ellipse_sigma=0, pointlabels=None):
    from matplotlib import pyplot as plt

    from .vectorops import split_by_categories, dummycode

    fig = plt.figure()

    def get_markers():
        colors = ["red", "blue", "green", "orange", "black"]
        markers = ["o", 7, "D", "x"]
        mrk = []
        for m in markers:
            for c in colors:
                mrk.append((c, m))
        return mrk

    def construct_confidence_ellipse(x, y):
        from matplotlib.patches import Ellipse

        vals, vecs = np.linalg.eig(np.cov(x, y))

        w = np.sqrt(vals[0]) * ellipse_sigma * 2
        h = np.sqrt(vals[1]) * ellipse_sigma * 2
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=w, height=h, angle=theta)
        ell.set_facecolor("none")
        ell.set_edgecolor(color)
        ax.add_artist(ell)

    def scat3d(Xs):
        x, y, z = Xs.T
        plt.scatter(x=x, y=y, zs=z, zdir="z", c=color,
                    marker=marker, label=translate(ct))

    def scat2d(Xs):
        x, y = Xs.T

        if ellipse_sigma:
            construct_confidence_ellipse(x, y)

        plt.scatter(x=x, y=y, c=color, marker=marker,
                    label=translate(ct))

    if points.shape[-1] == 3:
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D
        mode = "3d"
        ax = fig.add_subplot(111, projection="3d")
        scat = scat3d
    else:
        mode = "2d"
        ax = fig.add_subplot(111)
        scat = scat2d

    dependents, translate = dummycode(dependents)
    axlabels = axlabels[:int(mode[0])]

    by_categories = split_by_categories(points, dependents)
    setters = [ax.set_xlabel, ax.set_ylabel]
    if mode == "3d":
        setters.append(ax.set_zlabel)

    for st, axlb in zip(setters, axlabels):
        st(axlb)
    for ct, ctup in zip(by_categories, get_markers()):
        color, marker = ctup
        scat(by_categories[ct])

    if pointlabels is not None:
        for xy, lab in zip(points, pointlabels):
            ax.annotate(lab, xy, textcoords="data")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=7, mode="expand", borderaxespad=0.)
    plt.show()
