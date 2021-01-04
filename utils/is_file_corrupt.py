import h5py
import netCDF4 as nc


def is_hdf_corrupt(fn):
    try:
        h5_obj = h5py.File(fn)
    except:
        flag = True
    else:
        flag = False

    return flag


def is_nc_corrupt(fn):
    try:
        nc_obj = nc.Dataset(fn)
    except:
        flag = True
    else:
        flag = False

    return flag
