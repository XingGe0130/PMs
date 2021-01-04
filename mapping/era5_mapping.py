import numpy as np

from utils.loader import NCReader
from utils.bilinear import bilinear
from utils.date_handler import hours_to_time_stamp


class ERA5Mapping(object):

    def __init__(self):
        self.__dataset_dict = dict()
        self.__dataset_attrs_dict = dict()
        self.__valid_range_dict = {
            "blh": [8, 3500],
            "rh2m": [0, 100],
            "sp": [47000, 110000],
            "t2m": [225, 315],
            "u10": [-25, 30],
            "v10": [-25, 30]
        }
        self.desc = "mapping era5 to geo grid(10KM)"

    @staticmethod
    def calc_rh2m(t2m, d2m):
        es = 6.11 * 10.0 ** (7.5 * (t2m - 273.15) / (237.7 + (t2m - 273.15)))
        e = 6.11 * 10.0 ** (7.5 * (d2m - 273.15) / (237.7 + (d2m - 273.15)))
        rh = (e / es) * 100
        return rh

    def mapping(self, era5_fn, ymd, dst_lon, dst_lat, dataset_name_list=None):
        if dataset_name_list is None:
            dataset_name_list = ["blh", "rh2m", "sp", "t2m", "u10", "v10"]
        nc_reader = NCReader(era5_fn)

        hours = nc_reader.get_dataset("time")
        hours = hours.astype(np.float)
        time_stamp_lst = [hours_to_time_stamp(t) for t in hours]
        flag = [True if ts.startswith(ymd) else False for ts in time_stamp_lst]

        for dataset_name in dataset_name_list:

            if dataset_name == "rh2m":
                t2m = self.mask_scale("t2m", flag, nc_reader)
                t2m_daily = self.to_daily(t2m)

                d2m = self.mask_scale("d2m", flag, nc_reader)
                d2m_daily = self.to_daily(d2m)

                daily_data = self.calc_rh2m(t2m_daily, d2m_daily)
            else:
                data = self.mask_scale(dataset_name, flag, nc_reader)
                daily_data = self.to_daily(data)

            warped_data = self.up_sample(daily_data, dst_lon, dst_lat)

            self.__dataset_dict[dataset_name] = warped_data
            self.__dataset_attrs_dict[dataset_name] = {
                "long_name": nc_reader.get_dataset_attr(dataset_name, "long_name"),
                "units": nc_reader.get_dataset_attr(dataset_name, "units"),
                "FillValue": -32767,
                "Slope": 1,
                "Intercept": 0,
                "valid_range": self.__valid_range_dict.get(dataset_name)
            }

        return self.__dataset_dict, self.__dataset_attrs_dict

    @staticmethod
    def mask_scale(dataset_name, flag, nc_reader):
        data = nc_reader.get_dataset(dataset_name)[flag, :, :].astype(np.float)
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
    def up_sample(daily_data, dst_lon, dst_lat):

        # warped_data = bilinear(daily_data, src_shape, dst_shape)

        H, W = daily_data.shape

        left_top_x = 72.
        left_top_y = 55.
        grid_dot_per_degree = 4
        L = np.floor((left_top_y - dst_lat) * grid_dot_per_degree).astype(int)
        P = np.floor((dst_lon - left_top_x) * grid_dot_per_degree).astype(int)
        mask = L >= 0
        mask &= L < H
        mask &= P >= 0
        mask &= P < W
        gbl_pixel = P[mask]
        gbl_line = L[mask]
        warped_data = np.full(dst_lat.shape, -32767, 'f8')
        warped_data[mask] = daily_data[gbl_line, gbl_pixel]

        return warped_data

    @staticmethod
    def to_daily(data):

        mean_data = np.nanmean(data, axis=0)
        mean_data = np.where(np.isnan(mean_data), -32767, mean_data)

        return mean_data
