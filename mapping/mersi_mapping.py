import numpy as np

from utils.loader import HDFReader


class MERSIMapping(object):

    def __init__(self):
        self.__dataset_dict = dict()
        self.__dataset_attr_dict = dict()
        self.__valid_range_dict = {"AOT_550_Mean": [0, 1]}
        self.desc = "mapping FY3-D MERSI AOD to geo grid 10km"

    def mapping(self, mersi_fn, dataset_name, dst_lon, dst_lat):
        h5_reader = HDFReader(mersi_fn)
        data = self.mask_scale(dataset_name, h5_reader)

        warped_data = self.down_sample(data, dst_lat, dst_lon)

        self.__dataset_dict[dataset_name] = warped_data
        self.__dataset_attr_dict[dataset_name] = {
            "long_name": h5_reader.get_dataset_attr(dataset_name, "long_name"),
            "units": h5_reader.get_dataset_attr(dataset_name, "units"),
            "Slope": 1,
            "Intercept": 0,
            "FillValue": -32767,
            "valid_range": self.__valid_range_dict.get(dataset_name)
        }

        return self.__dataset_dict, self.__dataset_attr_dict

    @staticmethod
    def down_sample(data, dst_lat, dst_lon):
        H = data.shape[0]
        W = data.shape[1]

        left_top_x = -180.
        left_top_y = 90
        grid_dot_per_degree = 20
        L = np.floor((left_top_y - dst_lat) * grid_dot_per_degree).astype(int)
        P = np.floor((dst_lon - left_top_x) * grid_dot_per_degree).astype(int)
        mask = L >= 0
        mask &= L < H
        mask &= P >= 0
        mask &= P < W
        gbl_pixel = P[mask]
        gbl_line = L[mask]
        warped_data = np.full(dst_lat.shape, -32767, 'f8')
        warped_data[mask] = data[gbl_line, gbl_pixel]
        # print(warped_data)

        return warped_data

    @staticmethod
    def mask_scale(dataset_name, h5_reader):
        data = h5_reader.get_dataset(dataset_name)
        data = data.astype(np.float)
        fill_value = h5_reader.get_dataset_attr(dataset_name, "FillValue")
        slope = h5_reader.get_dataset_attr(dataset_name, "Slope")
        intercept = h5_reader.get_dataset_attr(dataset_name, "Intercept")
        # min_val, max_val = h5_reader.get_dataset_attr(dataset_name, "valid_range")

        mask = data == fill_value
        mask |= data <= 0
        mask |= data >= 1000
        data[~mask] = data[~mask] * slope + intercept
        data[mask] = -32767
        return data
