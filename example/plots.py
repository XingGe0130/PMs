import os

os.environ["PROJ_LIB"] = "/home/shkdeve/hm/software/miniconda3/envs/bupj/share/basemap/"
import glob

import numpy as np
import pandas as pd

from plot.PMLine import PMLine
from plot.PMScatter import PMScatter
from plot.PMMap import PMMap
from plot.PMHist import PMHist
from utils.loader import HDFReader


def get_roi_latlon(area_extent=None, res=0.1):
    if area_extent is None:
        area_extent = [34, 109, 43, 120]
    x = np.linspace(area_extent[1], area_extent[3], int(((area_extent[3] - area_extent[1]) / res) + 1))
    y = np.linspace(area_extent[0], area_extent[2], int(((area_extent[2] - area_extent[0]) / res) + 1))
    lon, lat = np.meshgrid(x, y)
    dst_lon = np.round(lon, 2)
    dst_lat = np.round(lat, 2)
    return dst_lat, dst_lon


def get_roi_mean_df(actual, gbl_line, gbl_pixel, hdf_list, pred):
    df = pd.DataFrame(columns=["date", actual, pred])
    for idx, f in enumerate(sorted(hdf_list)):
        hdf_reader = HDFReader(f)
        df_c = pd.DataFrame(columns=["date", actual, pred])
        date_str = os.path.basename(f)[:8]
        fill_value = hdf_reader.get_dataset_attr(actual, "FillValue")
        pm25 = hdf_reader.get_dataset(actual)[gbl_line, gbl_pixel]
        pm25[pm25 == fill_value] = np.nan
        pred_pm25 = hdf_reader.get_dataset(pred)[gbl_line, gbl_pixel]
        pred_pm25[pred_pm25 == fill_value] = np.nan
        df_c.loc[idx, "date"] = date_str
        df_c.loc[idx, actual] = np.nanmean(pm25)
        df_c.loc[idx, pred] = np.nanmean(pred_pm25)

        df = df.append(df_c, ignore_index=True)
        df = df.dropna()
    return df


def get_roi_rcs(dst_lat, dst_lon):
    H, W = (521, 641)
    left_top_x = 72.
    left_top_y = 55
    grid_dot_per_degree = 10
    L = np.floor((left_top_y - dst_lat) * grid_dot_per_degree).astype(int)
    P = np.floor((dst_lon - left_top_x) * grid_dot_per_degree).astype(int)
    mask = L >= 0
    mask &= L < H
    mask &= P >= 0
    mask &= P < W
    gbl_pixel = P[mask]
    gbl_line = L[mask]
    return gbl_line, gbl_pixel


def get_season_year_mean_data_dict(hdf_lst, dataset_name):
    result_dict = dict()
    season_fp_dict = dict()
    for season in ["spring", "summer", "autumn", "winter"]:
        season_fp_dict[season] = list()
    for f in hdf_lst:
        mon = str(int(os.path.basename(f)[4:6]))
        if str(mon) in ["3", "4", "5"]:
            season_fp_dict["spring"].append(f)
        elif str(mon) in ["6", "7", "8"]:
            season_fp_dict["summer"].append(f)
        elif str(mon) in ["9", "10", "11"]:
            season_fp_dict["autumn"].append(f)
        else:
            season_fp_dict["winter"].append(f)

    year_data_lst = list()
    for key in season_fp_dict.keys():
        season_data_lst = list()
        for f in season_fp_dict.get(key):
            hdf_reader = HDFReader(f)
            pred = hdf_reader.get_dataset(dataset_name)
            fill_value = hdf_reader.get_dataset_attr(dataset_name, "FillValue")
            pred[pred == fill_value] = np.nan
            season_data_lst.append(pred)

        season_data = np.asarray(season_data_lst)
        season_mean = np.nanmean(season_data, axis=0)
        season_mean = np.flipud(season_mean)
        result_dict[key] = season_mean
        year_data_lst.append(season_mean)

    year_data = np.asarray(year_data_lst)
    year_mean = np.nanmean(year_data, axis=0)
    result_dict["year"] = year_mean
    return result_dict


def line_plot(config_dict):
    fig_fp = config_dict.get("fig_fp")
    data_fp = config_dict.get("data_fp")

    actual = config_dict.get("actual")
    pred = config_dict.get("pred")
    area_extent = config_dict.get("area_extent")
    res = config_dict.get("res")
    title = config_dict.get("title")
    x_label = config_dict.get("x_label")
    y_label = config_dict.get("y_label")

    hdf_list = glob.glob(os.path.join(data_fp, "*.HDF"))

    dst_lat, dst_lon = get_roi_latlon(area_extent, res)

    gbl_line, gbl_pixel = get_roi_rcs(dst_lat, dst_lon)

    df = get_roi_mean_df(actual, gbl_line, gbl_pixel, hdf_list, pred)

    plot_line = PMLine()
    plot_line.draw(df, title, x_label, y_label, fig_fp)


def day_scatter_plot(config_dict):
    fig_fp = config_dict.get("fig_fp")
    os.makedirs(os.path.basename(fig_fp), exist_ok=True)
    hdf_fp = config_dict.get("hdf_fp")

    actual = config_dict.get("actual")
    pred = config_dict.get("pred")
    area_extent = config_dict.get("area_extent")
    res = config_dict.get("res")
    title = config_dict.get("title")
    x_label = config_dict.get("x_label")
    y_label = config_dict.get("y_label")

    dst_lat, dst_lon = get_roi_latlon(area_extent, res)

    gbl_line, gbl_pixel = get_roi_rcs(dst_lat, dst_lon)

    hdf_reader = HDFReader(hdf_fp)
    pred_data = hdf_reader.get_dataset(pred)
    actual_data = hdf_reader.get_dataset(actual)
    fill_val = hdf_reader.get_dataset_attr(pred, "FillValue")

    actual_data = actual_data[gbl_line, gbl_pixel]
    pred_data = pred_data[gbl_line, gbl_pixel]

    mask = actual_data != fill_val
    mask &= pred_data != fill_val

    pred_data = pred_data[mask]
    actual_data = actual_data[mask]

    pm_scatter = PMScatter()
    pm_scatter.draw(actual_data, pred_data, title, x_label, y_label, fig_fp)


def multi_day_scatter_plot(config_dict):
    fig_fd = config_dict.get("fig_fd")
    hdf_fd = config_dict.get("hdf_fd")

    actual = config_dict.get("actual")
    pred = config_dict.get("pred")
    area_extent = config_dict.get("area_extent")
    res = config_dict.get("res")
    title = config_dict.get("title")
    x_label = config_dict.get("x_label")
    y_label = config_dict.get("y_label")

    os.makedirs(fig_fd, exist_ok=True)
    hdf_list = glob.glob(os.path.join(hdf_fd, "*.HDF"))

    dst_lat, dst_lon = get_roi_latlon(area_extent, res)

    gbl_line, gbl_pixel = get_roi_rcs(dst_lat, dst_lon)

    pre_data_dict = get_season_year_mean_data_dict(hdf_list, pred)
    actual_data_dict = get_season_year_mean_data_dict(hdf_list, actual)

    for key in actual_data_dict.keys():
        pred_data = pre_data_dict.get(key)
        actual_data = actual_data_dict.get(key)

        pred_data = pred_data[gbl_line, gbl_pixel]
        actual_data = actual_data[gbl_line, gbl_pixel]

        mask = np.where(~(np.isnan(pred_data) | np.isnan(actual_data)))

        pred_data = pred_data[mask]
        actual_data = actual_data[mask]

        pm_scatter = PMScatter()
        pm_scatter.draw(actual_data, pred_data, title,
                        x_label, y_label, os.path.join(fig_fd, key + ".png"))


def day_map_plot(config_dict):
    hdf_fn = config_dict.get("hdf_fn")
    fig_fp = config_dict.get("fig_fp")
    area_extent = config_dict.get("area_extent")
    dataset_name = config_dict.get("dataset_name")
    colorbar_title = config_dict.get("colorbar_title")
    title = config_dict.get("title")

    llcrnrlon = area_extent[1]
    llcrnrlat = area_extent[0]
    urcrnrlon = area_extent[3]
    urcrnrlat = area_extent[2]
    projection = "cyl"
    lon_0 = ((area_extent[3] - area_extent[1]) / 2) + area_extent[1]
    lat_0 = ((area_extent[2] - area_extent[0]) / 2) + area_extent[0]

    hdf_reader = HDFReader(hdf_fn)
    pred = hdf_reader.get_dataset(dataset_name)
    fill_value = hdf_reader.get_dataset_attr(dataset_name, "FillValue")
    pred[pred == fill_value] = np.nan
    pred = np.flipud(pred)

    map_drawer = PMMap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, projection, lon_0, lat_0)
    map_drawer.add_data2D(pred, colorbar_title, title)

    map_drawer.save_fig(fig_fp)


def multi_day_map_plot(config_dict):
    hdf_fd = config_dict.get("hdf_fd")
    fig_fd = config_dict.get("fig_fd")
    area_extent = config_dict.get("area_extent")
    dataset_name = config_dict.get("dataset_name")
    colorbar_title = config_dict.get("colorbar_title")
    title = config_dict.get("title")

    llcrnrlon = area_extent[1]
    llcrnrlat = area_extent[0]
    urcrnrlon = area_extent[3]
    urcrnrlat = area_extent[2]
    projection = "cyl"
    lon_0 = ((area_extent[3] - area_extent[1]) / 2) + area_extent[1]
    lat_0 = ((area_extent[2] - area_extent[0]) / 2) + area_extent[0]

    os.makedirs(fig_fd, exist_ok=True)
    data_dict = get_season_year_mean_data_dict(sorted(glob.glob(os.path.join(hdf_fd, "*.HDF"))), dataset_name)
    for key in data_dict.keys():
        data = data_dict.get(key)
        map_drawer = PMMap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, projection, lon_0, lat_0)
        map_drawer.add_data2D(data, colorbar_title, title)
        map_drawer.save_fig(os.path.join(fig_fd, key + ".png"))


def hist_plot(config_dict):
    fig_fp = config_dict.get("fig_fp")
    data_fp = config_dict.get("data_fp")

    actual = config_dict.get("actual")
    pred = config_dict.get("pred")
    area_extent = config_dict.get("area_extent")
    res = config_dict.get("res")
    title = config_dict.get("title")
    x_label = config_dict.get("x_label")
    y_label = config_dict.get("y_label")

    hdf_list = glob.glob(os.path.join(data_fp, "*.HDF"))

    dst_lat, dst_lon = get_roi_latlon(area_extent, res)

    gbl_line, gbl_pixel = get_roi_rcs(dst_lat, dst_lon)

    df = get_roi_mean_df(actual, gbl_line, gbl_pixel, hdf_list, pred)

    pm_hist = PMHist()
    pm_hist.draw(df.loc[:, "PM2.5"] - df.loc[:, "Pred_PM2.5"], title,
                 x_label, y_label, range=(-30, 30), out_fn=fig_fp)


if __name__ == '__main__':
    map_dict = {
        "hdf_fd": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\result_PM2.5",
        "fig_fd": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\spatial",
        "area_extent": [3, 72, 55, 136],
        "dataset_name": "Pred_PM2.5",
        "colorbar_title": "Concentration(units: $ug/m^3$)",
        "title": "PM$_{2.5}$ Spatial Distribution",
    }
    multi_day_map_plot(map_dict)

    scatter_dict = {
        "hdf_fp": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\result_PM2.5\20181218.HDF",
        "fig_fp": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\spatial\scatter_20181218.png",
        "area_extent": [3, 72, 55, 136],
        "res": 0.1,
        "pred": "Pred_PM2.5",
        "actual": "PM2.5",
        "colorbar_title": "PM$_{2.5}$ VS Pred_PM$_{2.5}$",
        "x_label": "PM$_{2.5}$($ug/m^3$)",
        "y_label": "Pred_PM$_{2.5}$($ug/m^3$)",
    }
    # line_plot()
    day_scatter_plot(scatter_dict)
    config = {
        "hdf_fd": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\result_PM2.5",
        "fig_fd": r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\scatter",
        "area_extent": [3, 72, 55, 136],
        "res": 0.1,
        "pred": "Pred_PM2.5",
        "actual": "PM2.5",
        "colorbar_title": "PM$_{2.5}$ VS Pred_PM$_{2.5}$",
        "x_label": "PM$_{2.5}$($ug/m^3$)",
        "y_label": "Pred_PM$_{2.5}$($ug/m^3$)",
    }
    multi_day_scatter_plot(config)
