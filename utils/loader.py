import h5py
import pickle
import pandas as pd
import netCDF4 as nc


class HDFReader(object):

    def __init__(self, fn):
        self.__h5_obj = h5py.File(fn, "r")

    def get_dataset(self, dataset_name):
        data = self.__h5_obj.get(dataset_name)[:]
        return data

    def get_dataset_attr(self, dataset_name, attr_name):
        attr_val = self.__h5_obj.get(dataset_name).attrs.get(attr_name)
        return attr_val

    def get_global_attrs(self):
        global_attrs_dict = {key: self.__h5_obj.attrs.get(key) for key in self.__h5_obj.attrs.keys()}
        return global_attrs_dict

    def get_dataset_attrs(self, dataset_name):
        dataset_attrs_dict = {key: self.get_dataset_attr(dataset_name, key) for key in self.__h5_obj.get(dataset_name).attrs.keys()}
        return dataset_attrs_dict

    def __del__(self):

        try:
            self.__h5_obj.close()
        except:
            pass


class NCReader(object):

    def __init__(self, fn):
        self.__nc_obj = nc.Dataset(fn, 'r')
        self.__nc_obj.set_auto_maskandscale(False)

    def get_dataset(self, dataset_name):
        data = self.__nc_obj.variables[dataset_name][:]
        return data

    def get_dataset_attr(self, dataset_name, attr_name):
        try:
            attr_val = self.__nc_obj.variables[dataset_name].getncattr(attr_name)
        except:
            attr_val = "none"
        return attr_val

    def get_global_attrs(self):
        global_attrs_dict = {key: self.__nc_obj.getncattr(key) for key in self.__nc_obj.ncattrs()}
        return global_attrs_dict

    def get_dataset_attrs(self, dataset_name):
        dataset_attrs_dict = {key: self.get_dataset_attr(dataset_name, key) for key in self.__nc_obj.variables[dataset_name].ncattrs()}
        return dataset_attrs_dict

    def __del__(self):
        try:
            self.__nc_obj.close()
        except:
            pass


class CSVReader(object):

    def __init__(self, fn, encoding='utf-8'):
        self.__fn = fn
        self.__encoding = encoding

    def get_dataset(self):
        df = pd.read_csv(self.__fn, encoding=self.__encoding)
        return df


def pkl_loader(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data
