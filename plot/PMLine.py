import matplotlib.pyplot as plt


class PMLine(object):

    def __init__(self, pic_size=(40, 10), dpi=300):
        self.__pic_size = pic_size
        self.__pic_dpi = dpi
        self.fig, self.ax = plt.subplots(figsize=pic_size, dpi=dpi, facecolor="white", edgecolor="white")

    def set_frame(self, color="k", width=1.5):
        for spine in ["top", "bottom", "left", "right"]:
            self.ax.spines[spine].set_color(color)
            self.ax.spines[spine].set_linewidth(width)

    def draw(self, plot_df, title, x_label, y_label, out_fn=None):

        x = plot_df.loc[:, "date"]
        for col_name, col_val in plot_df.iteritems():
            if col_name == "date":
                continue
            self.ax.plot(x, col_val.values, label=col_name)

        self.ax.tick_params(labelsize=13)
        for label in self.ax.get_xmajorticklabels():
            label.set_horizontalalignment("right")

        plt.gcf().autofmt_xdate()
        self.set_frame()
        self.ax.set_title(title, fontdict={"family": "SimHei", "size": 24})
        self.ax.set_xlabel(x_label, fontdict={"family": "SimHei", "size": 22})
        self.ax.set_ylabel(y_label, fontdict={"family": "SimHei", "size": 22})
        self.ax.legend(edgecolor="k", facecolor="white")

        if out_fn is None:
            plt.show()
        else:
            plt.savefig(out_fn, dpi=300, bbox_inches="tight")
            plt.close()
