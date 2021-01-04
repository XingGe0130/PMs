import numpy as np

from utils.loader import NCReader


class MERRA2Mapping(object):

    def __init__(self):
        self.__dataset_dict = dict()
        self.__dataset_attrs_dict = dict()
        self.__vr_dict = {
            "BCSMASS": [3e-13, 9e-08],
            "DUSMASS": [6e-15, 4e-06],
            "DUSMASS25": [2e-15, 9e-07],
            "OCSMASS": [1e-12, 2e-06],
            "SO4SMASS": [3e-12, 2e-07],
            "SSSMASS": [2e-15, 6e-06],
            "SSSMASS25": [2e-16, 5.5e-07],
        }
        self.desc = "mapping MERRA2 dataset to geo grid 10km"

    def mapping(self, merra2_fn, dst_lon, dst_lat, dataset_name_list=None):
        if dataset_name_list is None:
            dataset_name_list = ["DUSMASS", "SSSMASS", "DUSMASS25", "SSSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS"]

        nc_reader = NCReader(merra2_fn)

        for dataset_name in dataset_name_list:
            data = self.mask_scale(dataset_name, nc_reader)
            daily_data = self.to_daily(data)
            warped_data = self.up_sample(daily_data, dst_lon, dst_lat)

            self.__dataset_dict[dataset_name] = warped_data
            self.__dataset_attrs_dict[dataset_name] = {
                "long_name": nc_reader.get_dataset_attr(dataset_name, "long_name"),
                "units": nc_reader.get_dataset_attr(dataset_name, "units"),
                "FillValue": -32767,
                "Slope": 1,
                "Intercept": 0,
                "valid_range": self.__vr_dict.get(dataset_name)
            }

        return self.__dataset_dict, self.__dataset_attrs_dict

    @staticmethod
    def mask_scale(dataset_name, nc_reader):

        data = nc_reader.get_dataset(dataset_name).astype(np.float)
        z, h, w = data.shape
        fill_value = nc_reader.get_dataset_attr(dataset_name, "_FillValue")
        missing_value = nc_reader.get_dataset_attr(dataset_name, "missing_value")
        slope = nc_reader.get_dataset_attr(dataset_name, "scale_factor")
        intercept = nc_reader.get_dataset_attr(dataset_name, "add_offset")

        mask = data == fill_value
        mask |= data == missing_value
        data[~mask] = data[~mask] * slope + intercept

        if np.any(mask):
            data[mask] = np.nan
            for level, line, pixel in np.argwhere(np.isnan(data)):
                r_start = line - 1
                r_stop = line + 2
                c_start = pixel - 1
                c_stop = pixel + 2

                if line == 0:
                    r_start = 0

                if line == h - 1:
                    r_stop = h

                if pixel == 0:
                    c_start = 0

                if pixel == w - 1:
                    c_stop = w
                data[level, line, pixel] = np.nanmean(data[level, r_start:r_stop, c_start:c_stop])
        return data

    @staticmethod
    def to_daily(data):

        mean_data = np.nanmean(data, axis=0)
        mean_data = np.where(np.isnan(mean_data), -32767, mean_data)

        return mean_data

    @staticmethod
    def up_sample(daily_data, dst_lon, dst_lat):

        H, W = daily_data.shape

        left_top_x = -180.
        left_top_y = -90
        x_grid_dot_per_degree = 1.6
        y_grid_dot_per_degree = 2
        L = np.floor((dst_lat - left_top_y) * y_grid_dot_per_degree).astype(int)
        P = np.floor((dst_lon - left_top_x) * x_grid_dot_per_degree).astype(int)
        mask = L >= 0
        mask &= L < H
        mask &= P >= 0
        mask &= P < W
        gbl_pixel = P[mask]
        gbl_line = L[mask]
        warped_data = np.full(dst_lat.shape, -32767, 'f8')
        warped_data[mask] = daily_data[gbl_line, gbl_pixel]

        return warped_data
