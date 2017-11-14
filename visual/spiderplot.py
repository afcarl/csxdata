import numpy as np
from matplotlib import pyplot as plt

from ..utilities.vectorop import split_by_categories as _sbc


def _unwrap(*arrays):
    output = [a if isinstance(a, np.ndarray) else a.as_matrix() for a in arrays]
    return output[0] if len(arrays) == 1 else output


def plot_single(x, paramnames, show=True, take_mean=False, ax=None, fill=True, **pltarg):
    x = _unwrap(x)
    if ax is None:
        ax = plt.subplot(111, polar=True)
    x = x.mean(axis=0) if take_mean else x
    dims = len(x)
    angles = [n / dims * 2 * np.pi for n in range(dims)]

    angles, x = np.append(angles, angles[0]), np.append(x, x[0])
    ax.plot(angles, x, label=pltarg.get("label"))
    if fill:
        ax.fill(angles, x, alpha=0.1)
    if pltarg.get("title"):
        ax.set_title(pltarg["title"], y=1.08, size=10)
    cX = np.linspace(0, 2*np.pi, 360)
    cY = np.zeros_like(cX)
    ax.plot(cX, cY, "g--", alpha=.5)
    ax.plot(cX, cY+2, "r--", alpha=.5)
    ax.plot(cX, cY-2, "r--", alpha=.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(paramnames, color="grey", size=8)
    ax.set_rmin(-3)
    ax.set_rmax(3)
    ax.set_rticks([-2, 0, 2])
    if show:
        plt.show()


def split_overlap(X, Y, paramnames, fill=True):
    X, Y = _unwrap(X, Y)
    bycat = _sbc(Y, X)
    ax = plt.subplot(111, polar=True)
    for cat in bycat:
        plot_single(bycat[cat], paramnames, show=False, take_mean=True, ax=ax, label=cat, fill=fill)
    plt.legend(loc="lower center", bbox_to_anchor=(0., 1.05, 1, 0.2), ncol=5)
    plt.show()


def split_gridlike(X, Y, paramnames, ncols=2):
    X, Y = _unwrap(X, Y)
    bycat = _sbc(Y, X)
    nrows = len(bycat) // ncols + int(len(bycat) % ncols > 0)
    fig, axarr = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True), figsize=(12, 8))
    for cat, ax in zip(bycat, axarr.flat):
        plot_single(
            bycat[cat], paramnames, show=False, take_mean=True, ax=ax, fill=True, title=cat,
            yupper=3, ylower=-3
                    )
    for ax in axarr.flat[len(bycat)-len(axarr.flat):]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()
