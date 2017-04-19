import numpy as np


def correlation(X, names=None, alpha=0.05):
    """Inspects the Pearson's (linear) and Spearman's (ranked) correlations"""
    from scipy.stats import spearmanr, pearsonr
    from matplotlib import pyplot

    if X.ndim != 2:
        raise ValueError("Only matrices are supported! X.ndim = {}".format(X.ndim))

    def get_spearmanr(data):
        scr, sprb = spearmanr(data, axis=0)
        if X.shape[1] > 2:
            return scr, sprb < alpha
        scnew = np.ones((2, 2))
        scnew[0, 1] = scr
        scnew[1, 0] = scr
        prnew = np.zeros_like(scnew)
        prnew[0, 1] = sprb
        prnew[1, 0] = sprb
        return scnew, prnew < alpha

    def get_pearsons(data):
        m, n = data.shape
        pcs, pps = [], []
        for i in range(n):
            for j in range(n):
                pc, pp = pearsonr(data[:, i], data[:, j])
                pcs.append(pc)
                pps.append(pp)
        return np.array(pcs).reshape(n, n), np.array(pps).reshape(n, n) < alpha

    if isinstance(names, np.ndarray):
        names = names.tolist()

    pcorr, pprob = get_pearsons(X)
    scorr, sprob = get_spearmanr(X)

    fig, axes = pyplot.subplots(2, 2, gridspec_kw={"width_ratios": [2, 2]})
    mats = [[pcorr, scorr], [pprob, sprob]]
    titles = [["Pearson's R", "Spearman's R"], ["Significance"]*2]
    for rown, vec in enumerate(mats):
        for coln, mat in enumerate(vec):
            cax = axes[rown][coln].matshow(
                mat, interpolation="none", vmin=0, vmax=1, cmap=pyplot.cm.get_cmap("bwr"))
            axes[rown][coln].set_title(titles[rown][coln], y=1.12)
            axes[rown][coln].set_xticks(np.arange(len(names)))
            axes[rown][coln].set_yticks(np.arange(len(names)))
            axes[rown][coln].set_xticklabels(names, rotation="vertical")
            axes[rown][coln].set_yticklabels(names)
    fig.suptitle("Correlations\nN = {}".format(len(X))
                 + (" (!)" if len(X) < 500 else ""))
    mng = pyplot.get_current_fig_manager()
    mng.window.showMaximized()
    pyplot.tight_layout()
    frm = lambda d: "{:> .3f}".format(d)
    print("PEARSON'S CORRELATION:")
    print("\n".join(", ".join(map(frm, line)) for line in pcorr))
    print("SPEARMAN'S CORRELATION:")
    print("\n".join(", ".join(map(frm, line)) for line in scorr))

    pyplot.show()


def category_frequencies(Y):
    """Inspects the representedness of categories in Y"""
    from ..utilities.vectorops import stringeq

    print("-"*38)
    categ = list(set(Y))
    nums = []
    rates = []

    for cat in categ:
        eq = stringeq(Y, cat)
        num = np.sum(eq)
        rate = num / Y.shape[0]
        rates.append(rate)
        nums.append(num)

    for cat, num, rate in sorted(zip(categ, nums, rates), key=lambda x: x[1], reverse=True):
        print("{0:<20}    {1:>3}    {2:>7.2%}".format(cat, num, rate))

    print("-" * 38)
    print("{0:<20}    {1:>3}    {2:.2%}".format("SUM", Y.shape[0], sum(rates)))
