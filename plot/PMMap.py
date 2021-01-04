import cmaps
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


class PMMap(object):

    def __init__(self,
                 llcrnrlon=72,
                 llcrnrlat=3,
                 urcrnrlon=136,
                 urcrnrlat=55,
                 projection="cyl",
                 lon_0=104,
                 lat_0=29):
        self._map = Basemap(
            llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, projection=projection, lon_0=lon_0, lat_0=lat_0
        )

        self._map.drawparallels(np.arange(llcrnrlat, urcrnrlat, 15), labels=[1, 0, 0, 0], fontsize=8)
        self._map.drawmeridians(np.arange(llcrnrlon, urcrnrlon, 15), labels=[0, 0, 0, 1], fontsize=8)
        # self._map.drawcountries()
        self._map.drawcoastlines()

    def add_data2D(self, data, colorbar_label, title=None):
        x = np.linspace(self._map.llcrnrx, self._map.urcrnrx, data.shape[1])
        y = np.linspace(self._map.llcrnry, self._map.urcrnry, data.shape[0])

        xx, yy = np.meshgrid(x, y)
        # m = self._map.pcolormesh(xx, yy, data, shading='flat', cmap=cmaps.GMT_panoply, latlon=False)
        m = self._map.pcolormesh(xx, yy, data, shading='flat', cmap=plt.cm.get_cmap("jet"), latlon=False)
        # add colorbar
        cb = self._map.colorbar(m, "bottom", size="5%", pad="8%")
        cb.ax.tick_params(labelsize=8, direction='in', length=2, width=1)
        cb.set_label(colorbar_label, fontdict={'family': 'SimHei', 'size': 10})

        # add a title.
        ax = plt.gca()
        ax.tick_params(direction='in', length=6, width=2, labelsize=18)
        ax.set_title(title, fontdict={'family': 'SimHei', 'size': 16})

    def add_data1D(self, lons, lats):
        lons, lats = self._map(lons, lats)
        self._map.scatter(lons, lats, s=1, marker="o", facecolors='r', edgecolors='r')

    def show(self):
        plt.show()

    def save_fig(self, out_fn):
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        plt.close()
