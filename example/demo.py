import os
import glob
import tqdm
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model

from utils.loader import HDFReader
from utils.writer import HDFWriter
from dataset.data_prepare import ParsePMDataset

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def write_result_to_hdf(pred_data, pred_target, cur_fp, out_dir, out_vars):
    vars_dict = dict()
    vars_attrs_dict = dict()
    h5_reader = HDFReader(cur_fp)
    gbl_attrs = h5_reader.get_global_attrs()

    for key in out_vars:
        data = h5_reader.get_dataset(key)

        if key == "demo_flag":
            pred_data_attrs = h5_reader.get_dataset_attrs(pred_target)
            pred_data_attrs["long_name"] = pred_target + " from PMNet"
            vars_dict["Pred_" + pred_target] = pred_data
            vars_attrs_dict["Pred_" + pred_target] = pred_data_attrs

        vars_dict[key] = data
        vars_attrs_dict[key] = h5_reader.get_dataset_attrs(key)

    os.makedirs(out_dir, exist_ok=True)
    h5_writer = HDFWriter(os.path.join(out_dir, os.path.basename(cur_fp)))
    h5_writer.set_global_attrs(gbl_attrs)

    for key in vars_dict.keys():
        data = vars_dict.get(key)
        data_attrs = vars_attrs_dict.get(key)
        h5_writer.create_dataset(key, data_attrs, data, data.dtype)


def demo(config_dict):
    test_fp = config_dict.get("test_fp")
    result_fp = config_dict.get("result_fp")
    model_path = config_dict.get("model_path")

    # 参数
    batch_size = config_dict.get("batch_size")
    x_vars = config_dict.get("x_vars")
    y_label = config_dict.get("y_label")
    out_vars = config_dict.get("out_vars")

    # 加载模型
    K.clear_session()
    ops.reset_default_graph()
    model = load_model(model_path)

    for f in tqdm.tqdm(sorted(glob.glob(os.path.join(test_fp, "*.HDF"))), ascii=True, desc="Prediction"):
        # for f in sorted(glob.glob(os.path.join(test_fp, "*.HDF"))):
        # print(f)
        parser = ParsePMDataset(f, x_vars=x_vars)
        h, w = parser.shape
        hh, ww = np.meshgrid(np.arange(h), np.arange(w))
        hh_flat = hh.flatten()
        ww_flat = ww.flatten()
        mask = parser.flag_xy(hh_flat, ww_flat, "demo_flag") == 0
        hh_flat = hh_flat[mask]
        ww_flat = ww_flat[mask]
        count = len(hh_flat)

        if count == 0:
            continue

        result_arr = np.full((h, w), -32767, "f4")

        x = parser.get_xy(hh_flat, ww_flat, return_y=False)
        x = x[:, :, np.newaxis]
        # print(">>>>>>>>>>>>>", x.shape, "<<<<<<<<<<<<<")
        pred = model.predict(x)
        # print(">>>>>>>>>>>>>", pred.shape, "<<<<<<<<<<<<<")
        result_arr[hh_flat, ww_flat] = pred[:, 0]

        result_fp = result_fp.format(y_label)
        write_result_to_hdf(result_arr, y_label, f, result_fp, out_vars)
