##############################################################################
#
# This is the history of Michele Cappellari modifications
# to Daniel Foreman-Mackey corner_plot routine.
#
# V1.0.0: Included "like" and "xstry" optional inputs
#   to show individual points coloured by their likelihood
#   and to show all points tried by MCMC.
#   Michele Cappellari, Oxford, 12 January 2014
# V1.1.0: Added "init" keyword to simply initialize an empty window.
#   MC, Oxford, 10 October 2014
# V1.1.1: Check that input sizes of "extents" and "labels" match.
#   MC, Portsmouth, 15 October 2014
# V1.1.2: Only show unique dots in "xs" and "like", but properly include
#   duplicates in histograms. MC, Oxford, 4 November 2014
# V1.1.3: Allow for scaling of axes labels. 
#   Fix program stop when values are the same within numerical accuracy
#   but np.std(x) is not exactly zero. MC, Oxford, 10 December 2014
# V1.1.4: Fix potential program stop with few input values.
#   MC, Oxford, 20 November 2015
# V1.1.5: Included histo_bin_width(). MC, Oxford, 8 February 2017
# V1.1.6: Added `rasterized` keyword and default value based on points number.
#   Use plt.plot for try values. MC, Oxford, 25 January 2018
# V1.1.7: Changes definition of `extents` for consistency with Scipy's `bounds`.
#   MC, Oxford, 27 April 2018
#
##############################################################################


from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner_plot", "hist2d", "error_ellipse"]
__version__ = "0.0.6"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
    "Adrian Price-Whelan @adrn",
    "Brendon Brewer @eggplantbren",
    "Ekta Patel @ekta1224",
    "Emily Rice @emilurice",
    "Geoff Ryan @geoffryan",
    "Kyle Barbary @kbarbary",
    "Phil Marshall @drphilmarshall",
    "Pierre Gratier @pirg",
]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

from plotbin.sauron_colormap  import register_sauron_colormap

######################################################################

def corner_plot(xs, like=None, xstry=None, weights=None, labels=None,
                extents=None, truths=None, truth_color="steelblue", fignum=None,
                scale_hist=False, quantiles=[], verbose=True, init=False,
                plot_contours=True, plot_datapoints=True, fig=None, 
                labels_scaling=1, rasterized=None, **kwargs):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like (nsamples, ndim)
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.
        
    xstry : array_like (nsamples, ndim) (optional)
        Contains all tried parameter, not only the accepted moves.
        
    like : array_like (nsamples) (optional)
        Likelihood of each xs[j, :] set of parameters to be shown on the plot.

    weights : array_like (nsamples,)
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    labels : iterable (ndim,) (optional)
        A list of names for the dimensions.

    extents : 2-tuple of array_like (2, ndim)
        Lower and upper bounds on independent variables. 
        Each array must match the size ndim, e.g.,
        [(0., 10., 30., 25.), (1., 15, 40., 55.)].

    truths : iterable (ndim,) (optional)
        A list of reference values to indicate on the plots.

    truth_color : str (optional)
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool (optional)
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable (optional)
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool (optional)
        If true, print the values of the computed quantiles.

    plot_contours : bool (optional)
        Draw contours for dense regions of the plot.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    fig : matplotlib.Figure (optional)
        Overplot onto the provided figure object.

    """
    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
        
    if xstry is not None:
        xstry = xstry.T

    if labels is not None:  # note xs was already trasposed
        assert len(labels) == xs.shape[0], 'lengths of labels must match number of dimensions'

    if extents is not None:
        extents = np.array(extents).T
        assert len(extents) == xs.shape[0], 'lengths of extents must match number of dimensions'

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, 'weights must be 1-D'
        assert xs.shape[1] == weights.shape[0], 'lengths of weights must match number of samples'

    register_sauron_colormap()

    # backwards-compatibility
    plot_contours = kwargs.get("smooth", plot_contours)

    K = len(xs)
    if fig is None:
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.05 * factor  # size of top/right margin
        whspace = 0.05         # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim
        fig, axes = plt.subplots(K, K, figsize=(dim, dim), num=fignum)
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)
        if init:
            return fig
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

        # Check for parameters that never change.
        m = np.array([e[0] == e[1] for e in extents], dtype=bool)
        if np.any(m):
            raise ValueError(("It looks like the parameter(s) in column(s) "
                              "{0} have no dynamic range. Please provide an "
                              "`extent` argument.")
                             .format(", ".join(map("{0}".format,
                                                   np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        for i in range(len(extents)):
            try:
                emin, emax = extents[i]
            except TypeError:
                q = [0.5 - 0.5*extents[i], 0.5 + 0.5*extents[i]]
                extents[i] = quantile(xs[i], q, weights=weights)

    # Extract unique values to reduce the memory taken by the figure (MC)
    # The unique version is used for the scatter plots but not for the histograms
    if like is not None:
        likeUnique, w = np.unique(like, return_index=True)
        xsUnique = xs[:, w]

        w = np.argsort(likeUnique) # Sort to plot most likely values last (MC)
        mx = likeUnique[w[-1]]     # Maximum likelihood value so far
        likeUnique = likeUnique[w]
        xsUnique = xsUnique[:, w]

        if rasterized is None:
            rasterized = True if xsUnique.size > 1e4 else False

    for i, x in enumerate(xs):
        ax = axes[i, i]

        # Plot the histograms. Scott's rule for bin size
        # For a Normal distribution sigma=1.4826*MAD, so the robust version of
        # Scott (1979) rule: binsize = 3.49*sigma/n^{1/3} is binsize = 5.17*MAD/n^{1/3}

        # binsize = 5.17*np.median(np.abs(x - np.median(x))) / x.size**(1./3.)
        binsize = 3.49*np.std(x) / x.size**(1./3.)
        # binsize = histo_bin_width(x)
        if binsize > 0:
            nbins = int((extents[i][1] - extents[i][0])/binsize)
            # if nbins > 20:
            #     nbins //= 2  # divide by two for aesthetic reasons and keep integer
            if nbins > 1e4:  # x values are the same within numerical accuracy
                nbins = 5
        else:
            nbins = 5
        ax.cla()  # clean current subplot before plotting
        n, b, p = ax.hist(x, weights=weights, bins=kwargs.get("bins", nbins),
                          range=extents[i], color=kwargs.get("color", "b"),
                          histtype='stepfilled')

        # Axes labels larger than tick labels
        ax.yaxis.label.set_size(plt.rcParams['font.size']*labels_scaling)
        ax.xaxis.label.set_size(plt.rcParams['font.size']*labels_scaling)

        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=kwargs.get("color", "k"))
            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs):
            ax = axes[i, j]

            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            ax.cla()  # clean current subplot before plotting

            ax.yaxis.label.set_size(plt.rcParams['font.size']*labels_scaling)
            ax.xaxis.label.set_size(plt.rcParams['font.size']*labels_scaling)

            if xstry is not None:
                ax.plot(xstry[j,:], xstry[i,:], ',m', ms=1, rasterized=rasterized, zorder=0)

            # Plot black when DeltaLogLike=9/2-->DeltaChi2=9 (3sigma)
            if like is not None:
                ax.scatter(xsUnique[j,:], xsUnique[i,:], c=likeUnique,
                           vmin=mx-4.5, vmax=mx, edgecolors='None',
                           cmap='sauron', rasterized=rasterized, **kwargs)
                ax.set_xlim(extents[j])
                ax.set_ylim(extents[i])
            else:
                hist2d(y, x, ax=ax, extent=[extents[j], extents[i]],
                       plot_contours=plot_contours,
                       plot_datapoints=plot_datapoints,
                       weights=weights, **kwargs)

#            ax.tick_params(width=2, which='major')

            if truths is not None:
                ax.plot(truths[j], truths[i], "s", color=truth_color)
                ax.axvline(truths[j], color=truth_color)
                ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig

######################################################################

def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()

######################################################################

def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot

######################################################################

def hist2d(x, y, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", plt.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 50)
    color = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", None)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)

    cmap = cm.get_cmap("gray")
    cmap._init()
    cmap._lut[:-3, :-1] = 0.
    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                 weights=kwargs.get('weights', None))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    V = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    X, Y = X[:-1], Y[:-1]

    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,
                rasterized=True)
        if plot_contours:
            ax.contourf(X1, Y1, H.T, [V[-1], H.max()],
                        cmap=LinearSegmentedColormap.from_list("cmap",
                                                               ([1] * 3,
                                                                [1] * 3),
                        N=2), antialiased=False)

    if plot_contours:
        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
        ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])

######################################################################

def _hist_bin_sturges(x):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return x.ptp() / (np.log2(x.size) + 1.0)

######################################################################

def _hist_bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

######################################################################

def histo_bin_width(x):
    """

    This function is copied from Numpy 1.11

    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x`. The Sturges estimator
    is quite good for small (<1000) datasets and is the default in the R
    language. This method gives good off the shelf behaviour.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    # There is no need to check for zero here. If ptp is, so is IQR and
    # vice versa. Either both are zero or neither one is.
    return min(_hist_bin_fd(x), _hist_bin_sturges(x))

######################################################################
