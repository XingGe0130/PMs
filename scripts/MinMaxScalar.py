import os
import glob

import tqdm
import json

from utils.loader import HDFReader
from utils.writer import HDFWriter


class MinMaxScalar(object):

    def __init__(self):
        self.vr_dict = {
            "BCSMASS": [float("inf"), -float("inf")],
            "DUSMASS": [float("inf"), -float("inf")],
            "DUSMASS25": [float("inf"), -float("inf")],
            "OCSMASS": [float("inf"), -float("inf")],
            "PM10": [float("inf"), -float("inf")],
            "PM2.5": [float("inf"), -float("inf")],
            "SO4SMASS": [float("inf"), -float("inf")],
            "SSSMASS": [float("inf"), -float("inf")],
            "SSSMASS25": [float("inf"), -float("inf")],
            "blh": [float("inf"), -float("inf")],
            "rh2m": [float("inf"), -float("inf")],
            "sp": [float("inf"), -float("inf")],
            "t2m": [float("inf"), -float("inf")],
            "u10": [float("inf"), -float("inf")],
            "v10": [float("inf"), -float("inf")]
        }

    def minmax(self, f):
        hdf_reader = HDFReader(f)
        for name in self.vr_dict.keys():
            data = hdf_reader.get_dataset(name)
            fill_val = hdf_reader.get_dataset_attr(name, "FillValue")

            if len(data[data != fill_val]) == 0:
                continue

            max_val = data[data != fill_val].max()
            min_val = data[data != fill_val].min()

            if max_val > self.vr_dict.get(name)[1]:
                self.vr_dict[name][1] = max_val

            if min_val < self.vr_dict.get(name)[0]:
                self.vr_dict[name][0] = min_val

    def to_json(self, fp):
        with open(fp, "w") as f:
            json.dump(self.vr_dict, f)


def json_loader(json_f):
    with open(json_f, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def write_scalar_to_json(f_lst, out_jf):
    scalar = MinMaxScalar()
    for f in tqdm.tqdm(f_lst, ascii=True, desc="minmax"):
        scalar.minmax(f)
    scalar.to_json(out_jf)


def dataset_format(ds_fst, scalar_f, out_fd):
    os.makedirs(out_fd, exist_ok=True)
    vr_dict = json_loader(scalar_f)

    for f in tqdm.tqdm(ds_fst, ascii=True, desc="dataset format"):
        hdf_reader = HDFReader(f)
        gbl_attrs = hdf_reader.get_global_attrs()

        hdf_writer = HDFWriter(os.path.join(out_fd, os.path.basename(f)))
        hdf_writer.set_global_attrs(gbl_attrs)

        names = hdf_reader.get_dataset_names()
        for name in names:
            data = hdf_reader.get_dataset(name)
            attrs = hdf_reader.get_dataset_attrs(name)
            if name in vr_dict.keys():
                attrs["valid_range"] = vr_dict.get(name)

            hdf_writer.create_dataset(name, attrs, data, data.dtype)


if __name__ == '__main__':
    fd = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\data\nn_data_v2"
    out_json = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\valid_json.json"
    out_fd = r"D:\Work_Directory\bupengju\03-ParticulateMatter\PMs\tmp\nn_data"
    hdf_lst = sorted(glob.glob(os.path.join(fd, "*HDF")))
    # write_scalar_to_json(hdf_lst, out_json)
    dataset_format(hdf_lst, out_json, out_fd)
