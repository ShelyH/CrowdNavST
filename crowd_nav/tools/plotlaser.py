import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from lmfit import Model
from matplotlib import colors, ticker, cm

from itertools import chain
import scipy.interpolate


def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))


def get_normed_histo(data, bins, rang):
    hist, bin_edges = np.histogram(data, bins, range=rang)
    hist = np.divide(hist, np.sum(hist))
    return hist, bin_edges


def get_gauss_fit(hist, edges):
    gmodel = Model(gaussian)
    result = gmodel.fit(hist, x=edges[:-1], amp=1, cen=0, wid=2)
    return result


plt.style.use('./mythesis.mplstyle')
rnge = [-1.5, 1.5]
stp = 0.5
rng_x = np.linspace(rnge[0], rnge[1], 7)
xrange, yrange = np.meshgrid(rng_x, rng_x)

width_xz = np.zeros(xrange.shape).flatten()
idx = 0
dz = 0

rng_hist = [-4, 4]
for dx, dy in zip(xrange.flatten(), yrange.flatten()):
    frame = "-x_{:.1f}".format(dx) + "_y_{:.1f}".format(dz) + "_z_{:.1f}".format(dy)

    print(dx, dy, dz)
    df = np.load('output/cross_dist/dist{}.npz'.format(frame))

    dist_d = df['d']
    hist, bin_edges = get_normed_histo(dist_d, 2*20 + 1, rng_hist)
    result = get_gauss_fit(hist, bin_edges)

    # plotting individual histograms
    #plt.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), label='full set')
    #plt.plot(bin_edges[:-1], result.best_fit, 'purple', label='best fit', alpha=0.5)
    #plt.show()

    amp = result.params['amp']
    cen = result.params['cen']
    wid = result.params['wid']
    print(result.redchi)
    width_xz[idx] = wid

    idx += 1
    del result


width_xz = width_xz.reshape(xrange.shape)
plt.contourf(xrange, yrange, width_xz, cmap=cm.viridis, interpolation='bicubic')
cbar = plt.colorbar()
plt.xlabel("$\Delta x[cm]$")
plt.ylabel("$\Delta z[cm]$")
cbar.set_label("width [cm]")

plt.show()