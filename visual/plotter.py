import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utilities.vectorops import split_by_categories


def markerstream(colors=None, markers=None, mode="normal"):

    def _random_stream(cl, mk):
        from random import choice
        while 1:
            yield choice(cl), choice(mk)

    def _nonrandom_stream(cl, mk):
        while 1:
            for m in mk:
                for c in cl:
                    yield c, m

    def _shuffled_stream(cl, mk):
        from random import shuffle
        cl, mk = list(map(shuffle, (cl, mk)))
        return _nonrandom_stream(cl, mk)

    if colors is None:
        colors = ["red", "blue", "green", "orange", "black"]
    if markers is None:
        markers = ["o", 7, "D", "x"]

    stream = {"normal": _nonrandom_stream,
              "random": _random_stream,
              "shuffled": _shuffled_stream}[mode](colors, markers)
    return stream


class Plotter2D:

    def __init__(self, fig, points, labels, title=None,
                 axlabels=None):
        self.fig = fig  # type: Figure
        self.ax = self.fig.add_subplot(111)  # type: Axes
        self.ax.autoscale(tight=True)
        self.ax.set_title(title if title else "")
        self.X = points
        self.Y = labels

        self._sanity_check()

        self.mrk = markerstream()
        self.color = "black"
        self.marker = "."
        axl = axlabels if axlabels is not None else [None]*2
        self.ax.set_xlabel(axl[0])
        self.ax.set_ylabel(axl[1])

    def _sanity_check(self):
        if self.X.ndim != 2:
            raise AttributeError("Only matrices are supported!")
        if self.X.shape[1] > 3 or self.X.shape[1] < 2:
            raise RuntimeError("Only 2 or 3 dimensional plotting is supported!")

    def _step_ctup(self):
        self.color, self.marker = next(self.mrk)

    def _scatter2D(self, Xs, label=None, **kw):
        x, y = Xs.T
        self.ax.scatter(x=x, y=y, c=self.color, marker=self.marker, **kw)
        if label is not None:
            self.ax.annotate(label, xy=(x, y), xycoords="data")

    def _fit_ellipse(self, Xs, sigma):
        from matplotlib.patches import Ellipse

        x, y = Xs.T

        mux, muy = np.mean(x), np.mean(y)
        vals, vecs = np.linalg.eig(np.cov(x, y))

        w = np.sqrt(vals[0]) * sigma * 2
        h = np.sqrt(vals[1]) * sigma * 2
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        ell = Ellipse(xy=(mux, muy), width=w, height=h, angle=theta)
        ell.set_facecolor("none")
        ell.set_edgecolor(self.color)

        self.ax.add_patch(ell)
        # self.ax.autoscale(tight=False)

    def _final_touches(self, dumppath):
        self.fig.tight_layout()
        self.ax.grid(True)
        if dumppath:
            self.fig.savefig(dumppath)

    def split_scatter(self, center=False, dumppath=None, sigma=2, alpha=1., **kw):
        split = split_by_categories(self.X, self.Y, argsonly=True)
        for categ in split:
            self._step_ctup()
            arg = split[categ]
            Xs = np.copy(self.X[arg])
            if center:
                self._scatter2D(Xs.mean(axis=0), label=categ)

            self._scatter2D(Xs, alpha=alpha, **kw)
            self._fit_ellipse(Xs, sigma)

        self._final_touches(dumppath)

    def scatter(self, label_points=None, dumppath=None, sigma=0):
        self._scatter2D(self.X, label_points)
        if sigma:
            self._fit_ellipse(self.X, sigma)
        self._final_touches(dumppath)

    @staticmethod
    def add_legend(plt, loc=None, ncol=7):
        if loc is None:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
                       ncol=ncol, mode="expand", borderaxespad=0.)
        else:
            plt.legend(loc=loc, ncol=ncol)

    def add_trendline(self, *args, **kw):
        line = np.polyfit(self.X[0], self.X[1], 1)
        line = np.poly1d(line)
        self.ax.plot(self.X[0], line(self.X[0]), *args, **kw)
