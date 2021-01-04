import os
import glob
import shutil

import tqdm
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.loader import CSVReader, HDFReader, NCReader, pkl_loader
from utils.writer import pkl_writer


class DataPrepare(object):

    def __init__(self, h5_list, batch_size, flag_name, x_vars, y_vars):

        self.batch_size = batch_size
        df = pd.DataFrame(columns=["h5_fp", "rows", "cols", "counts"])

        for f in tqdm.tqdm(h5_list, ascii=True, desc="Gen idx"):
            parser = ParsePMDataset(f)
            h, w = parser.shape
            df_c = pd.DataFrame(columns=["h5_fp", "rows", "cols", "counts"])
            hh, ww = np.meshgrid(np.arange(h), np.arange(w))
            hh_flat = hh.flatten()
            ww_flat = ww.flatten()
            mask = parser.flag_xy(hh_flat, ww_flat, flag_name) == 0
            hh_flat = hh_flat[mask]
            ww_flat = ww_flat[mask]
            count = len(hh_flat)

            if count == 0:
                continue

            if self.batch_size > count:
                batch_size = count

            lens = int(np.ceil(count / batch_size))
            for i in range(lens):
                df_c.loc[i, "h5_fp"] = os.path.basename(f)
                if i == lens - 1:
                    df_c.loc[i, "rows"] = hh_flat[i * batch_size:]
                    df_c.loc[i, "cols"] = ww_flat[i * batch_size:]
                    df_c.loc[i, "counts"] = len(ww_flat[i * batch_size:])
                else:
                    df_c.loc[i, "rows"] = hh_flat[i * batch_size: (i + 1) * batch_size]
                    df_c.loc[i, "cols"] = ww_flat[i * batch_size: (i + 1) * batch_size]
                    df_c.loc[i, "counts"] = len(ww_flat[i * batch_size: (i + 1) * batch_size])
            df = df.append(df_c, ignore_index=True)

        self.__df = df
        self.__x_vars = x_vars
        self.__y_vars = y_vars

    def copy_file(self, in_fd, out_fd):
        os.makedirs(out_fd, exist_ok=True)
        for f in np.unique(self.__df.loc[:, "h5_fp"].values):
            in_fn = os.path.join(in_fd, f)
            out_fn = os.path.join(out_fd, f)
            shutil.copyfile(in_fn, out_fn)

    def write_idx_to_csv(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        # self.__df.to_csv(fn, index=False)

    def write_data_to_pkl(self, h5_fd, out_fn, x_normal=True, y_normal=False):
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        data = list()
        for i in tqdm.tqdm(range(len(self.__df)), ascii=True, desc="Gen pkl"):
            fp, r, c, _ = self.__df.loc[i]
            f = os.path.join(h5_fd, fp)
            parser = ParsePMDataset(f, self.__x_vars, self.__y_vars)
            x, y = parser.get_xy(r, c, x_normal=x_normal, y_normal=y_normal, return_y=True)
            data.append(np.hstack((x, y)))
        result = np.vstack(data)
        print(np.where(result == -32767))

        # pkl_writer(result, out_fn)


class ParsePMDataset(object):

    def __init__(self, fp, x_vars: list = None, y_vars: list = None):
        self.__fp = fp
        if x_vars is None:
            self.__x_vars = ["AOT_550_Mean", "blh", "rh2m", "t2m", "u10", "v10"]
        else:
            self.__x_vars = x_vars

        if y_vars is None:
            self.__y_vars = ["PM2.5"]
        else:
            self.__y_vars = y_vars

    @property
    def shape(self):
        with h5py.File(self.__fp, "r") as f:
            shape = f["latitude"].shape
        return shape

    def scale_x(self, ds_obj, scan_line_num, pixel_line_num, normal):
        fill_val = ds_obj.attrs["FillValue"]
        slope = ds_obj.attrs["Slope"]
        intercept = ds_obj.attrs["Intercept"]
        ds_arr = ds_obj[:][scan_line_num, pixel_line_num]
        ds_arr[ds_arr == fill_val] = np.nan

        result = ds_arr * slope + intercept

        if normal:
            valid_range = ds_obj.attrs["valid_range"]
            assert len(valid_range) > 0
            min_val, max_val = valid_range * slope + intercept
            result = (result - min_val) / (max_val - min_val) * 0.8 + 0.1

        # 3 X 3 window
        # h, w = ds_obj[:].shape
        # result = np.zeros((len(scan_line_num)))
        # for i in range(len(scan_line_num)):
        #     r = scan_line_num[i]
        #     c = pixel_line_num[i]
        #     r_start = r - 1
        #     r_stop = r + 2
        #     c_start = c - 1
        #     c_stop = c + 2
        #
        #     if r == 0:
        #         r_start = 0
        #
        #     if r == h - 1:
        #         r_stop = h
        #
        #     if c == 0:
        #         c_start = 0
        #
        #     if c == w - 1:
        #         c_stop = w
        #
        #     data = np.nanmean(ds_arr[r_start:r_stop, c_start:c_stop])
        #     data = data * slope + intercept
        #
        #     if normal:
        #         valid_range = ds_obj.attrs["valid_range"]
        #         assert len(valid_range) > 0
        #         min_val, max_val = valid_range * slope + intercept
        #         data = (data - min_val) / (max_val - min_val) * 0.8 + 0.1
        #
        #     result[i] = data

        return result

    @staticmethod
    def scale_y(ds_obj, scan_line_num, pixel_line_num, normal):

        ds_arr = ds_obj[:][scan_line_num, pixel_line_num]

        slope = ds_obj.attrs["Slope"]
        intercept = ds_obj.attrs["Intercept"]

        result = ds_arr * slope + intercept

        if normal:
            valid_range = ds_obj.attrs["valid_range"]
            assert len(valid_range) == 2
            min_val, max_val = valid_range * slope + intercept
            result = (result - min_val) / (max_val - min_val) * 0.8 + 0.1

        return result

    def get_xy(self, scan_line_num, pixel_line_num, return_y=True, x_normal=True, y_normal=False):
        x = np.zeros((len(scan_line_num), len(self.__x_vars)))

        for i, x_var in enumerate(self.__x_vars):
            with h5py.File(self.__fp, "r") as f:
                x[:, i] = self.scale_x(f[x_var], scan_line_num, pixel_line_num, x_normal)
        if return_y:
            y = np.zeros((len(scan_line_num), len(self.__y_vars)))
            for i, y_var in enumerate(self.__y_vars):
                with h5py.File(self.__fp, "r") as f:
                    y[:, i] = self.scale_y(f[y_var], scan_line_num, pixel_line_num, y_normal)
            return x, y
        else:
            return x

    def flag_xy(self, scan_line_num, pixel_line_num, flag_name="train_flag"):
        with h5py.File(self.__fp, "r") as f:
            value = f[flag_name][:][scan_line_num, pixel_line_num]
        return value


def test_HDFReader():
    hdf_fn = r"D:\01-work_directory\03-PM2.5\PMs\data\MERSI\FY3D_MERSI_GBAL_L2_AOD_MLT_GLL_20180817_POAD_5000M_MS.HDF"
    h_reader = HDFReader(hdf_fn)
    data = h_reader.get_dataset("AOT_550_Mean")
    slope = h_reader.get_dataset_attr("AOT_550_Mean", "Slope")
    global_attrs = h_reader.get_global_attrs()
    print(global_attrs)
    aod_attrs = h_reader.get_dataset_attrs("AOT_550_Mean")
    print(aod_attrs)
    print(data.shape)
    print(slope)


def test_NCReader():
    nc_fn = r"/hxqtmp/DPLearning/bupj/PMs/data/MERRA2_400.tavg1_2d_aer_Nx.20180801.nc4"

    x = np.linspace(72, 136, 641)
    y = np.linspace(55, 3, 521)
    lon, lat = np.meshgrid(x, y)
    dst_lon = np.round(lon, 2)
    dst_lat = np.round(lat, 2)

    nc_reader = NCReader(nc_fn)
    nc_lat= nc_reader.get_dataset("lat")
    print(nc_lat)
    scale_factor = nc_reader.get_dataset_attr("DUANGSTR", "scale_factor")
    add_offset = nc_reader.get_dataset_attr("DUANGSTR", "add_offset")
    DUSMASS25 = nc_reader.get_dataset("DUANGSTR")
    DUSMASS25 = DUSMASS25[0, :, :]
    DUSMASS25 = DUSMASS25 * scale_factor + add_offset
    H, W = DUSMASS25.shape
    print(H, W)

    left_top_x = -180.
    left_top_y = 90
    x_grid_dot_per_degree = 1.6
    y_grid_dot_per_degree = 2
    L = np.floor((left_top_y - dst_lat) * y_grid_dot_per_degree).astype(int)
    P = np.floor((dst_lon - left_top_x) * x_grid_dot_per_degree).astype(int)
    mask = L >= 0
    mask &= L < H
    mask &= P >= 0
    mask &= P < W
    gbl_pixel = P[mask]
    gbl_line = L[mask]
    print(gbl_line)
    print(gbl_pixel)
    warped_data = np.full(dst_lat.shape, -32767, 'f8')
    warped_data[mask] = DUSMASS25[gbl_line, gbl_pixel]
    print(np.where(warped_data==-32767))

    plt.figure()
    plt.imshow(DUSMASS25)
    plt.show()

    plt.figure()
    plt.imshow(warped_data)
    plt.show()


def test_CSVReader():
    site_info_fn = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\data\aq\站点列表-2019.08.01起.csv"
    site_data_fn = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\data\aq\data\china_sites_20180818.csv"
    site_info_reader = CSVReader(site_info_fn, 'UTF-8')
    site_info_df = site_info_reader.get_dataset()
    # print(site_info_df)

    site_data_reader = CSVReader(site_data_fn, 'UTF-8')
    site_data_df = site_data_reader.get_dataset()
    # print(site_data_df)
    daily_data_df = site_data_df[(site_data_df["hour"] == 15) & (
            (site_data_df["type"] == "PM2.5_24h") | (site_data_df["type"] == "PM10_24h"))]
    print(pd.isna(daily_data_df[daily_data_df["type"] == "PM10_24h"].loc[:, "1408A"]))


def test_pkl_loader():
    fn = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\data\seg_data\pm25\pkl\train_data.pkl"
    data = pkl_loader(fn)
    print(data.shape)

    for i in range(data.shape[0]):
        print(data[i, -1])

    # data = data[:, :, np.newaxis]
    # print(data.shape)
    # data = data[0:1024, :-1, :]
    # print(data.shape)


def test_MERRA2_mapping():
    from mapping.merra2_mapping import MERRA2Mapping
    from utils.writer import HDFWriter

    x = np.linspace(72, 136, 641)
    y = np.linspace(55, 3, 521)
    lon, lat = np.meshgrid(x, y)
    dst_lon = np.round(lon, 2)
    lat_attr = {
        "long_name": "latitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [3, 55]
    }
    dst_lat = np.round(lat, 2)
    lon_attr = {
        "long_name": "longitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [72, 136]
    }

    merra2_fn = r"/hxqtmp/DPLearning/bupj/PMs/data/MERRA2_400.tavg1_2d_aer_Nx.20180801.nc4"

    merra2_mapping = MERRA2Mapping()
    merra2_dataset_dict, merra2_dataset_attr_dict = merra2_mapping.mapping(merra2_fn, dst_lon, dst_lat)

    h5_writer = HDFWriter(r"/hxqtmp/DPLearning/bupj/PMs/tmp/merra2.HDF")
    h5_writer.create_dataset("longitude", lon_attr, dst_lon, lon.dtype)
    h5_writer.create_dataset("latitude", lat_attr, dst_lat, lat.dtype)
    for merra2_name in merra2_dataset_dict.keys():
        merra2_data = merra2_dataset_dict.get(merra2_name)
        merra2_attr = merra2_dataset_attr_dict.get(merra2_name)
        h5_writer.create_dataset(merra2_name, merra2_attr, merra2_data, merra2_data.dtype)


def test_era5_mapping():
    from mapping.era5_mapping import  ERA5Mapping
    from utils.writer import HDFWriter

    x = np.linspace(72, 136, 641)
    y = np.linspace(55, 3, 521)
    lon, lat = np.meshgrid(x, y)
    dst_lon = np.round(lon, 2)
    lat_attr = {
        "long_name": "latitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [3, 55]
    }
    dst_lat = np.round(lat, 2)
    lon_attr = {
        "long_name": "longitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [72, 136]
    }

    era5_fn = r"/hxqtmp/DPLearning/bupj/PMs/data/MERRA2_400.tavg1_2d_aer_Nx.20180801.nc4"

    era5_mapping = ERA5Mapping()
    era5_dataset_dict, era5_dataset_attr_dict = era5_mapping.mapping(era5_fn, dst_lon, dst_lat)

    h5_writer = HDFWriter(r"/hxqtmp/DPLearning/bupj/PMs/tmp/era5.HDF")
    h5_writer.create_dataset("longitude", lon_attr, dst_lon, lon.dtype)
    h5_writer.create_dataset("latitude", lat_attr, dst_lat, lat.dtype)
    for era5_name in era5_dataset_dict.keys():
        era5_data = era5_dataset_dict.get(era5_name)
        era5_attr = era5_dataset_attr_dict.get(era5_name)
        h5_writer.create_dataset(era5_name, era5_attr, era5_data, era5_data.dtype)


def test_nn_data():
    batch_size = 4096
    x_vars = ["AOT_550_Mean", "BCSMASS", "DUSMASS25", "OCSMASS", "SO4SMASS", "SSSMASS25", "blh", "rh2m", "t2m", "u10", "v10"]
    y_vars = ["PM2.5"]
    hdf_fp = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\data\nn_data"
    valid_files = sorted(glob.glob(os.path.join(hdf_fp, "*.HDF")))
    valid_dp = DataPrepare(valid_files, batch_size, "train_flag", x_vars, y_vars)
    # valid_dp.copy_file(data_fp, os.path.join(out_fp, "hdf", "valid"))
    # valid_dp.write_idx_to_csv(os.path.join(os.path.dirname(valid_pkl), "valid_idx.csv"))
    valid_dp.write_data_to_pkl(hdf_fp, "./valid_pkl", x_normal=False)


if __name__ == '__main__':
    # test_HDFReader()
    # test_NCReader()
    # test_CSVReader()
    test_pkl_loader()
    # test_MERRA2_mapping()
    # test_nn_data()
