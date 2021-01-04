import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import make_interp_spline


class PMHist(object):

    def __init__(self, pic_size=(15, 10), dpi=300):
        self.__pic_size = pic_size
        self.__pic_dpi = dpi
        self.fig, self.ax = plt.subplots(figsize=pic_size, dpi=dpi, facecolor="white", edgecolor="white")

    def set_frame(self, color="k", width=1.5):
        for spine in ["top", "bottom", "left", "right"]:
            self.ax.spines[spine].set_color(color)
            self.ax.spines[spine].set_linewidth(width)

    def add_annotation(self, s, loc='upper left', pad=0.4, borderpad=0., prop=None):
        if prop is None:
            prop = {"size": 18, "color": "k"}
        anchored_text = AnchoredText(
            s, loc=loc, pad=pad, borderpad=borderpad, frameon=False, prop=prop
        )
        self.ax.add_artist(anchored_text)

    def draw(self, diff_data, title, x_label, y_label, nbins=150, range=(-5, 5), out_fn=None):
        n, bins, patches = self.ax.hist(diff_data, nbins, range=range, density=True, color="white", edgecolor="k",
                                        rwidth=1, align="left")

        # add a smooth line
        x_smooth = np.linspace(bins.min(), bins.max(), 300)
        y_smooth = make_interp_spline(bins[:-1], n)(x_smooth)

        self.ax.plot(x_smooth, y_smooth, c="k", linestyle="--", zorder=2)
        self.ax.tick_params(direction='in', length=6, width=2, labelsize=18)
        self.ax.set_xlim(range)
        self.ax.set_xlabel(x_label, fontdict={"size": 18, "color": "k"})
        self.ax.set_ylabel(y_label, fontdict={"size": 18, "color": "k"})
        self.ax.set_title(title, fontdict={"size": 20, "color": "k", }, pad=20)
        self.set_frame()

        # Tweak spacing to prevent clipping of y label
        self.fig.tight_layout()
        if out_fn is None:
            plt.show()
        else:
            plt.savefig(out_fn, dpi=300, bbox_inches='tight')
            plt.close()
