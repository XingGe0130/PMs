import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from utils.loader import CSVReader


class AirQualityMapping(object):

    def __init__(self):
        self.__dataset_dict = dict()
        self.__dataset_attr_dict = dict()
        self.__valid_range_dict = {"PM2.5": [0, 1031.5], "PM10": [0, 3916]}
        self.desc = "mapping station to geo grid(10KM)"

    @staticmethod
    def get_site_data(site_data_fn, encoding="utf-8"):
        csv_reader = CSVReader(site_data_fn, encoding=encoding)
        site_data = csv_reader.get_dataset()
        return site_data

    def mapping(self, geo_grid_dots, site_info_fn, site_data_fn):
        try:
            site_info_df = self.get_site_data(site_info_fn)
        except UnicodeDecodeError:
            site_info_df = self.get_site_data(site_info_fn, encoding="gbk")

        try:
            site_data_df = self.get_site_data(site_data_fn)
        except UnicodeDecodeError:
            site_data_df = self.get_site_data(site_data_fn, encoding="gbk")

        lon, lat = geo_grid_dots

        try:
            daily_data_df = site_data_df[(site_data_df["hour"] == 15) & (
                    (site_data_df["type"] == "PM2.5_24h") | (site_data_df["type"] == "PM10_24h"))]
        except KeyError:
            daily_data_df = pd.DataFrame()

        if not daily_data_df.empty:
            pm10, pm25 = self.to_grid(daily_data_df, site_info_df, lat, lon)

        else:
            pm25, pm10 = np.full(lon.shape, -32767, "f8"), np.full(lon.shape, -32767, "f8")

        self.__dataset_dict["PM2.5"] = pm25
        self.__dataset_dict["PM10"] = pm10
        self.__dataset_attr_dict["PM2.5"] = {
            "long_name": "PM2.5 from station",
            "units": "ug/m3",
            "Slope": 1,
            "Intercept": 0,
            "FillValue": -32767,
            "valid_range": self.__valid_range_dict.get("PM2.5")
        }
        self.__dataset_attr_dict["PM10"] = {
            "long_name": "PM10 from station",
            "units": "ug/m3",
            "Slope": 1,
            "Intercept": 0,
            "FillValue": -32767,
            "valid_range": self.__valid_range_dict.get("PM10")
        }

        return self.__dataset_dict, self.__dataset_attr_dict

    @staticmethod
    def to_grid(daily_data_df, site_info_df, lat, lon):
        pm25, pm10, count = np.zeros(lon.shape), np.zeros(lon.shape), np.zeros(lon.shape)
        tree = KDTree(list(zip(lon.ravel(), lat.ravel())))
        for site_id in daily_data_df.columns.values[3:]:
            pts = site_info_df[site_info_df["监测点编码"] == site_id][["经度", "纬度"]].values.flatten()

            if len(pts) == 0:
                continue

            if len(pts) > 2:
                pts = pts[:2]

            p25 = daily_data_df[daily_data_df["type"] == "PM2.5_24h"][site_id].values.item()
            p10 = daily_data_df[daily_data_df["type"] == "PM10_24h"][site_id].values.item()
            if (not pd.isna(p25)) and (not pd.isna(p10)):
                dist_arr, idx_arr = tree.query(pts)
                target_pts = tree.data[idx_arr]
                row, col = np.argwhere((lon == target_pts[0]) & (lat == target_pts[1])).flatten()
                pm25[row, col] += p25
                pm10[row, col] += p10
                count[row, col] += 1
        mask = count != 0
        pm25[mask] /= count[mask]
        pm10[mask] /= count[mask]
        pm25[~mask] = -32767
        pm10[~mask] = -32767
        return pm10, pm25
