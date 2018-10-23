import matplotlib as plt
import numpy as np


def plot_fpred(fpred, xlim=(-5, 5), ylim=(-5, 5)):

    xmn, xmx = xlim
    ymn, ymx = ylim

    XX, YY = np.meshgrid(np.arange(xmn, xmx, 0.05),
                         np.arange(ymn, ymx, 0.05))

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = fpred(np.c_[XX.ravel(), XX.ravel()])
    Z = Z.reshape(XX.shape)

    f = plt.figure()
    ax = plt.gca()
    ax.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu)

    return f


if __name__ == '__main__':
    print('foo')
