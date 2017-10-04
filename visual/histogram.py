from scipy import stats
from matplotlib import pyplot as plt, mlab


class PlotBase:

    def __init__(self, x, ax=None):
        self.x = x
        self.mu, self.sigma = x.mean(), x.std()
        self.ax = plt.gca() if ax is None else ax


class Histogram(PlotBase):

    def plot(self, bins=20, normed=1, axtitle="Histogram"):
        n, bins, patches = self.ax.hist(self.x, bins, normed=normed, facecolor="green",
                                        edgecolor="black", alpha=0.75)
        if normed:
            y = mlab.normpdf(bins, self.mu, self.sigma)
            self.ax.plot(bins, y, "r--", linewidth=1)

        self.ax.grid(True)
        self.ax.set_title(axtitle)

        return self.ax


class NormProb(PlotBase):

    def plot(self, axtitle="Normal Probability Plot"):
        stats.probplot(self.x, plot=self.ax)
        self.ax.set_title(axtitle)

        return self.ax


def fullplot(x, paramname, histbins=20):
    fig, ax = plt.subplots(1, 2)
    Histogram(x, ax[0]).plot(histbins)
    NormProb(x, ax[1]).plot()
    plt.suptitle(paramname)
    plt.show()
