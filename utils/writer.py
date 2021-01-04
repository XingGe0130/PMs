import h5py
import pickle


class HDFWriter(object):
    def __init__(self, filename):
        self.__h5_obj = h5py.File(filename, 'w')

    def __del__(self, ):
        try:
            self.__h5_obj.close()
        except:
            pass

    def set_global_attrs(self, attr_dict: dict):
        for key, value in attr_dict.items():
            self.__h5_obj.attrs[key] = value

    def create_group(self, group_name, attributes: dict = None):
        group = self.__h5_obj.create_group(group_name)

        if attributes:
            for key, value in attributes.items():
                group_name.attrs[key] = value

        return group

    def create_dataset(self, dataset_name, attributes, data, dtype, parent=None):
        if parent is None:
            parent = self.__h5_obj

        ds = parent.create_dataset(dataset_name, data=data, dtype=dtype, compression='gzip', compression_opts=9)

        for key, value in attributes.items():
            ds.attrs[key] = value


def pkl_writer(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)
