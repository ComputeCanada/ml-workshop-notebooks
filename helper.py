from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg
import numpy.random as npr
from scipy.special import expit


def gen_planar_samples(
        *, complexity=10, noisiness=0.33, num=256, xylim=(-5, 5)):
    '''
    Generates random 2D features and labels based on plane waves.
    '''

    # hack to work around late binding
    def xsin(amp, k, phi, xy):
        return amp * np.sin(k @ xy + phi)

    funcs = []
    for i in range(complexity):
        # wavevector, factor out scale
        k = npr.uniform(i / 2, i + 1, size=(2,)) / xylim[1]
        k *= (2 * (npr.random(size=(2,)) - 0.5))
        # phase
        phi = npr.uniform(0, 2 * np.pi)
        # amplitude
        amp = xylim[1] / nlg.norm(k)
        # amp = 1

        funcs.append(partial(xsin, amp, k, phi))

    def amplitude(xys):
        def amp_row(xy):
            return sum(f(xy) for f in funcs)

        amp = np.apply_along_axis(amp_row, 1, xys)
        amp -= amp.mean()
        amp /= np.std(amp)
        amp = expit(amp)

        return amp

    xys = npr.uniform(xylim[0], xylim[1], size=(num, 2))
    amp = amplitude(xys)

    # will work without size, but will be WRONG
    ythresh = noisiness * npr.random(size=xys.shape[:1])

    y = np.zeros(len(xys))
    y[amp > 0.5 * (1 - noisiness) + ythresh] = 1

    return xys, y, amplitude


def plot_decision_surface(fpred, xlim=(-5, 5), ylim=(-5, 5)):

    xmn, xmx = xlim
    ymn, ymx = ylim

    XX, YY = np.meshgrid(np.arange(xmn, xmx, 0.05),
                         np.arange(ymn, ymx, 0.05))

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = fpred(np.c_[XX.ravel(), YY.ravel()])

    # specialcase two classes, dirty but worx
    if len(Z.shape) == 2 and Z.shape[1] == 2:
        Z = Z[:, 0]

    # print(np.c_[XX.ravel(), XX.ravel()].shape)
    # Z = np.apply_along_axis(fpred, 1, np.c_[XX.ravel(), XX.ravel()])
    Z = Z.reshape(XX.shape)

    # f = plt.figure()
    ax = plt.gca()
    ax.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, vmin=0, vmax=1)

    # return f


if __name__ == '__main__':
    neg, pos, y, ampfun = gen_planar_samples()
    plt.scatter(neg[:, 0], neg[:, 1], color='red')
    plt.scatter(pos[:, 0], pos[:, 1], color='blue')
    plot_decision_surface(ampfun)
    plt.show()
