import os
import shutil

import tqdm
import h5py
import numpy as np
import pandas as pd

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
        self.__df.to_csv(fn, index=False)

    def write_data_to_pkl(self, h5_fd, out_fn, x_normal=True, y_normal=False, return_y=True):
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        data = list()
        for i in tqdm.tqdm(range(len(self.__df)), ascii=True, desc="Gen pkl"):
            fp, r, c, _ = self.__df.loc[i]
            f = os.path.join(h5_fd, fp)
            parser = ParsePMDataset(f, self.__x_vars, self.__y_vars)
            x, y = parser.get_xy(r, c, x_normal=x_normal, y_normal=y_normal, return_y=return_y)
            data.append(np.hstack((x, y)))
        result = np.vstack(data)

        pkl_writer(result, out_fn)


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


def split_files_by_mon(f_lst):
    date_fp_dict = dict()
    for mon in range(1, 13):
        date_fp_dict[str(mon)] = list()
    for f in f_lst:
        mon = str(int(os.path.basename(f)[4:6]))
        date_fp_dict[mon].append(f)
    train_files = list()
    valid_files = list()
    test_files = list()
    np.random.seed(42)
    for key in date_fp_dict.keys():
        fp_lst = np.asarray(date_fp_dict.get(key))
        train_idx = np.random.choice(len(fp_lst), int(np.ceil(len(fp_lst) * 0.85)), replace=False)
        train_files.extend(fp_lst[train_idx].tolist())
        fp_lst = np.delete(fp_lst, train_idx)
        valid_idx = np.random.choice(len(fp_lst), int(np.ceil(len(fp_lst) * 0.5)), replace=False)
        valid_files.extend(fp_lst[valid_idx].tolist())
        test_files.extend(np.delete(fp_lst, valid_idx))
    return train_files, valid_files, test_files
