"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""
import warnings

import numpy as np


class RandomClassifierMock:

    """Mocks a most basic interface of an sklearn classifier"""

    def __init__(self):
        self.Y = None

    # noinspection PyUnusedLocal
    def fit(self, X, Y):
        self.Y = np.unique(Y)

    def predict(self, X):
        """Returns random unweighted class 'predictions'"""
        return np.random.choice(self.Y, len(X))


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
