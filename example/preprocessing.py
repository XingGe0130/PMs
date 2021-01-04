import os
import glob
import datetime

import tqdm
import numpy as np

from utils.writer import HDFWriter
from utils.date_handler import subtract_day
from utils.is_file_corrupt import is_nc_corrupt
from utils.is_file_corrupt import is_hdf_corrupt
from mapping.era5_mapping import ERA5Mapping
from mapping.mersi_mapping import MERSIMapping
from mapping.merra2_mapping import MERRA2Mapping
from mapping.air_quality_mapping import AirQualityMapping


def preprocessing(config_dict):
    era5_fp = config_dict.get("era5_fp")
    merra2_fp = config_dict.get("merra2_fp")
    out_p = config_dict.get("out_p")
    mersi_fp = config_dict.get("mersi_fp")
    site_data_fn = config_dict.get("site_data_fn")
    site_info_fn = config_dict.get("site_info_fn")
    area_extent = config_dict.get("area_extent")
    res = config_dict.get("res")

    x = np.linspace(area_extent[1], area_extent[3], int(((area_extent[3] - area_extent[1]) / res) + 1))
    y = np.linspace(area_extent[2], area_extent[0], int(((area_extent[2] - area_extent[0]) / res) + 1))
    lon, lat = np.meshgrid(x, y)
    lon = np.round(lon, 2)
    lon_attr = {
        "long_name": "longitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [72, 136]
    }

    lat = np.round(lat, 2)
    lat_attr = {
        "long_name": "latitude",
        "units": "degree",
        "FillValue": -32767,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": [3, 55]
    }

    os.makedirs(out_p, exist_ok=True)

    for f in tqdm.tqdm(sorted(glob.glob(os.path.join(site_data_fn, "*.csv"))), ascii=True, desc="gen nn data"):
        fn = os.path.basename(f)
        utc_date = subtract_day(fn.split("_")[2][:8])

        if not np.all([os.path.exists(era5_fp.format(utc_date[:6])), os.path.exists(mersi_fp.format(utc_date)),
                       os.path.exists(merra2_fp.format(utc_date))]):
            continue

        if np.any([is_nc_corrupt(era5_fp.format(utc_date[:6])), is_nc_corrupt(merra2_fp.format(utc_date)),
                   is_hdf_corrupt(mersi_fp.format(utc_date))]):
            continue

        train_flag = np.ones(lon.shape).astype(np.bool)
        demo_flag = np.ones(lon.shape).astype(np.bool)

        air_quality_mapping = AirQualityMapping()
        aq_dataset_dict, aq_dataset_attr_dict = air_quality_mapping.mapping([lon, lat], site_info_fn, f)

        era5_mapping = ERA5Mapping()
        era5_dataset_dict, era5_dataset_attr_dict = era5_mapping.mapping(era5_fp.format(utc_date[:6]), utc_date,
                                                                         lon, lat)

        merra2_mapping = MERRA2Mapping()
        merra2_dataset_dict, merra2_dataset_attr_dict = merra2_mapping.mapping(merra2_fp.format(utc_date), lon, lat)

        mersi_mapping = MERSIMapping()
        mersi_dataset_dict, mersi_dataset_attr_dict = mersi_mapping.mapping(mersi_fp.format(utc_date), "AOT_550_Mean",
                                                                            lon, lat)

        write_to_hdf(aq_dataset_attr_dict, aq_dataset_dict, era5_dataset_attr_dict, era5_dataset_dict, train_flag,
                     demo_flag, merra2_dataset_dict, merra2_dataset_attr_dict, mersi_dataset_dict,
                     mersi_dataset_attr_dict, lon, lon_attr, lat, lat_attr, out_p,
                     utc_date)


def write_to_hdf(aq_dataset_attr_dict, aq_dataset_dict, era5_dataset_attr_dict, era5_dataset_dict, train_flag,
                 demo_flag, merra2_dataset_dict, merra2_dataset_attr_dict, mersi_dataset_dict, mersi_dataset_attr_dict,
                 lon, lon_attr, lat, lat_attr, out_p, utc_date):
    h5_writer = HDFWriter(os.path.join(out_p, utc_date + ".HDF"))
    h5_writer.set_global_attrs({
        "File Name": utc_date + ".HDF",
        "Resolution X": 0.1,
        "Resolution Y": 0.1,
        "Left-Bottom X": 72,
        "Left-Bottom Y": 3,
        "Left-Top X": 72,
        "Left-Top Y": 55,
        "Right-Bottom X": 136,
        "Right-Bottom Y": 3,
        "Right-Top X": 136,
        "Right-Top Y": 55,
        "Projection Type": "latlon",
        "Data Creating DateTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '(Beijing Time)',
    })

    h5_writer.create_dataset("longitude", lon_attr, lon, lon.dtype)
    h5_writer.create_dataset("latitude", lat_attr, lat, lat.dtype)

    train_flag = np.ones(lon.shape).astype(bool)
    demo_flag = np.ones(lon.shape).astype(bool)
    for aq_name in aq_dataset_dict.keys():
        aq_data = aq_dataset_dict.get(aq_name)
        aq_attr = aq_dataset_attr_dict.get(aq_name)
        train_flag &= aq_data != -32767
        h5_writer.create_dataset(aq_name, aq_attr, aq_data, aq_data.dtype)

    for era5_name in era5_dataset_dict.keys():
        era5_data = era5_dataset_dict.get(era5_name)
        era5_attr = era5_dataset_attr_dict.get(era5_name)

        train_flag &= era5_data != -32767
        demo_flag &= era5_data != -32767
        h5_writer.create_dataset(era5_name, era5_attr, era5_data, era5_data.dtype)

    for merra2_name in merra2_dataset_dict.keys():
        merra2_data = merra2_dataset_dict.get(merra2_name)
        merra2_attr = merra2_dataset_attr_dict.get(merra2_name)

        train_flag &= merra2_data != -32767
        demo_flag &= merra2_data != -32767
        h5_writer.create_dataset(merra2_name, merra2_attr, merra2_data, merra2_data.dtype)

    for mersi_name in mersi_dataset_dict.keys():
        mersi_data = mersi_dataset_dict.get(mersi_name)
        mersi_attr = mersi_dataset_attr_dict.get(mersi_name)

        train_flag &= mersi_data != -32767
        demo_flag &= mersi_data != -32767
        h5_writer.create_dataset(mersi_name, mersi_attr, mersi_data, mersi_data.dtype)

    train_flag = train_flag.astype(np.int)
    train_flag = np.where(train_flag == 0, -1, 0)

    train_flag_attr = {
        "long_name": "valid data flag for train",
        "units": "none",
        "FillValue": -1,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": 0
    }
    h5_writer.create_dataset("train_flag", train_flag_attr, train_flag, train_flag.dtype)

    demo_flag = demo_flag.astype(np.int)
    demo_flag = np.where(demo_flag == 0, -1, 0)

    demo_flag_attr = {
        "long_name": "valid data flag for demo",
        "units": "none",
        "FillValue": -1,
        "Slope": 1,
        "Intercept": 0,
        "valid_range": 0
    }
    h5_writer.create_dataset("demo_flag", demo_flag_attr, demo_flag, demo_flag.dtype)
