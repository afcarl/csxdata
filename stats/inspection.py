import numpy as np


def correlation(X, names=None, alpha=0.05):
    """Inspects the Pearson's (linear) and Spearman's (ranked) correlations"""
    from scipy.stats import spearmanr
    from matplotlib import pyplot

    def get_pearsons_ps(pcorrel):
        from scipy.stats import betai
        df = pcorrel.shape[0]
        ix, iy = np.diag_indices_from(pcorrel)
        pcorrel[ix, iy] -= 1e-7
        t_sq = pcorrel**2 * (df / ((1.0 - pcorrel) * (1.0 + pcorrel)))
        return betai(0.5*df, 0.5, df / (df + t_sq))

    if isinstance(names, np.ndarray):
        names = names.tolist()

    pcorr = np.abs(np.corrcoef(X, rowvar=0))
    pprob = np.less_equal(get_pearsons_ps(pcorr), alpha).astype(int)
    scorr, sprob = np.abs(spearmanr(X, axis=0))
    sprob = np.less_equal(sprob, alpha).astype(int)

    fig, axes = pyplot.subplots(2, 2, gridspec_kw={"width_ratios": [2, 2]})
    mats = [[pcorr, scorr], [pprob, sprob]]
    titles = [["Pearson's R", "Spearman's R"], ["Significance"]*2]
    for rown, vec in enumerate(mats):
        for coln, m in enumerate(vec):
            axes[rown][coln].matshow(m, interpolation="none", vmin=0, vmax=1)
            axes[rown][coln].set_title(titles[rown][coln], y=1.12)
            axes[rown][coln].set_xticks(np.arange(len(names)))
            axes[rown][coln].set_yticks(np.arange(len(names)))
            axes[rown][coln].set_xticklabels(names, rotation="vertical")
            axes[rown][coln].set_yticklabels(names)
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
