import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confmatrix(X: np.ndarray, Y: np.ndarray, model, dumpdata=None, dumpplot=None, show=True, **kw):
    onehot = Y.ndim == 2
    y = Y.argmax(axis=1) if onehot else Y.copy()
    categories = np.sort(np.unique(y))
    ncat = len(categories)
    predictions = model.predict(X)
    if onehot:
        predictions = predictions.argmax(axis=1)
    cm = confusion_matrix(y, predictions, categories).T
    ax = plt.gca()
    ax.imshow(cm)
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    for i in range(ncat):
        for j in range(ncat):
            plt.text(i, j, f"{cm[i, j]}", horizontalalignment="center")
    plt.tight_layout()
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.title(kw.get("title", ""))
    if dumpplot:
        plt.savefig(dumpplot)
    if dumpdata is not None:
        pd.DataFrame(data=cm, index=pd.Index(sorted(categories), name="True labels"), columns=sorted(categories)
                     ).to_excel(dumpdata)
    if show:
        plt.show()
    else:
        plt.clf()
    return cm
