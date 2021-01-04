import numpy as np
from scipy import optimize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


class PMScatter(object):

    def __init__(self, pic_size=(15, 10), dpi=300):
        self.__pic_size = pic_size
        self.__pic_dpi = dpi
        self.fig, self.ax = plt.subplots(figsize=pic_size, dpi=dpi, facecolor="white", edgecolor="white")

    @staticmethod
    def __func(x, a, b):
        return a * x + b

    def __ploy_fit(self, x, y):
        slope, intercept = optimize.curve_fit(self.__func, x, y)[0]
        return slope, intercept

    def add_fit_line(self, x=None, y=None, c="k", linewidth=1.5, linestyle="-"):

        # plot fit line
        slope, intercept = self.__ploy_fit(x, y)
        fit_line = slope * x + intercept
        lines = self.ax.plot(x, fit_line, c=c, linewidth=linewidth, linestyle=linestyle)
        return lines, slope, intercept

    def add_1_to_1_line(self, x=None, c="r", linewidth=1.5, linestyle="-"):

        lines = self.ax.plot(x, x, c=c, linewidth=linewidth, linestyle=linestyle)
        return lines

    def add_heat_map(self, x, y, bins=300, cmap=cm.get_cmap("jet"), vmin=0,
                     vmax=100):

        # Compute the bi-dimensional histogram of two data samples.
        H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

        # H need to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)

        # mask pixels with a value of zero
        Hmasked = np.ma.masked_where(H == 0, H)

        ret = plt.pcolormesh(x_edges, y_edges, Hmasked, cmap=cmap, vmin=vmin, vmax=vmax)
        return ret

    def add_annotation(self, s, loc='upper left', pad=0.4, borderpad=0., prop=None):
        if prop is None:
            prop = {"size": 18, "color": "k"}
        anchored_text = AnchoredText(
            s, loc=loc, pad=pad, borderpad=borderpad, frameon=False, prop=prop
        )
        self.ax.add_artist(anchored_text)

    def set_frame(self, color="k", width=1.5):
        for spine in ["top", "bottom", "left", "right"]:
            self.ax.spines[spine].set_color(color)
            self.ax.spines[spine].set_linewidth(width)

    def draw(self, x, y, title, x_label, y_label, out_fn=None):
        self.ax.scatter(x, y)

        self.ax.tick_params(labelsize=13)
        self.add_heat_map(x, y)
        self.add_1_to_1_line(x)
        self.add_fit_line(x, y)
        self.set_frame()
        self.ax.set_xlim((x.min(), x.max()))
        self.ax.set_ylim((x.min(), x.max()))
        self.ax.tick_params(left=True, bottom=True, direction="in", labelsize=16, length=6, width=2)
        self.ax.set_title(title, fontdict={"family": "SimHei", "size": 24})
        self.ax.set_xlabel(x_label, fontdict={"family": "SimHei", "size": 22})
        self.ax.set_ylabel(y_label, fontdict={"family": "SimHei", "size": 22})

        if out_fn is None:
            plt.show()
        else:
            plt.savefig(out_fn, dpi=300, bbox_inches="tight")
            plt.close()
