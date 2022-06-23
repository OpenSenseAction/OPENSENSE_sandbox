# ugly WIP code, copy-paste from dev notebook

import os
import urllib
import zipfile
from functools import partial

import pandas as pd
import xarray as xr


def download_data_file(url, local_path=".", local_file_name=None, print_output=False):
    if not os.path.exists(local_path):
        if print_output:
            print(f"Creating path {local_path}")
        os.makedirs(local_path)

    if local_file_name is None:
        local_file_name = url.split("/")[-1]

    if os.path.exists(os.path.join(local_path, local_file_name)):
        print(
            f"File already exists at desired location {os.path.join(local_path, local_file_name)}"
        )
        print("Not downloading!")
        return

    if print_output:
        print(f"Downloading {url}")
        print(f"to {local_path}/{local_file_name}")

    request_return_meassage = urllib.request.urlretrieve(
        url, os.path.join(local_path, local_file_name)
    )
    return request_return_meassage



# Single Eband CML data from Czech Republic

download_fencl_2021_Eband_data = partial(
    download_data_file,
    url="https://zenodo.org/record/5013463/files/Dataset_1.0.0.zip",
)

def tranform_fencl_2021_Eband_data(fn):
    # open ZIP file
    with zipfile.ZipFile(fn) as zfile:
        # get file handle for CSV file stored in the ZIP file
        f = zfile.open("data_metedata/commercial_microwave_link_total_loss/cml_total_loss.csv")
        # Parse data from file
        df_data = pd.read_csv(
            f,
            index_col=0,
            parse_dates=True,
            sep=';',
        )
        # Build xarray.Dataset. Here we used hardcoded metadata (taken from the metadata
        # file in the ZIP file) since there is only one CML and since the metadata file
        # is hard to parse
        ds = xr.Dataset(
            data_vars={
                'trsl': (('sublink_id', 'time'), [df_data.s73.values, df_data.s83.values]),
                'frequency': (('sublink_id'), [73.5, 83.5]),
            },
            coords=dict(
                time=df_data.index.values,
                cml_id='cz_example_cml_1',
                length=4.866,
                sublink_id=(('sublink_id'), ['ab', 'ba']),
                site_a_longitude=14.53,
                site_b_longitude=14.53, # coordinates have been rounded for publication...
                site_a_latitude=50.03,
                site_b_latitude=50.03,
            ),
        )
        return ds
        

# Dutch CML data from https://data.4tu.nl/ndownloader/files/24025658

download_overeem_2019_large_CML_data_Netherlands = partial(
    download_data_file,
    url="https://data.4tu.nl/ndownloader/files/24025658",
    local_file_name="data.zip",
)

def transform_overeem_2019_large_CML_data_Netherlands(fn, nrows=None):
    # open ZIP file
    with zipfile.ZipFile(fn) as zfile:
        # get file handle for CSV file stored in the ZIP file
        f = zfile.open("CMLs_20120530_20120901.dat")

        # Read content of CSV file
        df2012 = pd.read_csv(
            f,
            nrows=nrows,  # if desired, do not read the full file here to save time
            sep="\s+",
            skiprows=1,
            names=[
                "frequency",
                "datetime",
                "pmin",
                "pmax",
                "pathlength",
                "xstart",
                "ystart",
                "xend",
                "yend",
                "id",
            ],
        )

    # set correct date index
    df2012 = df2012.set_index(pd.to_datetime(df2012["datetime"], format="%Y%m%d%H%M"))

    import numpy as np

    # This is the empty list where we will store the intermediate
    # `xarray.Dataset` for each CML which we will later concatenate
    # to one big `xarray.Dataset`
    ds_list = []
    all_date_time = xr.cftime_range(
        str(df2012.index.min()), str(df2012.index.max()), freq="15min"
    )
    for cml_id in np.unique(df2012.id):
        df_sel = df2012[df2012["id"] == cml_id]

        # identical cml_ids with duplicate date time entries probably
        # stem from incorrect metadata, these are printed here and discarded for now
        if df_sel.index.duplicated().sum() > 0:
            print(
                "cml "
                + str(cml_id)
                + " has "
                + str(df_sel.index.duplicated().sum())
                + " duplicated entries from "
                + str(len(df_sel))
                + " rows."
            )
        df_sel = df_sel[~df_sel.index.duplicated()]

        # This combines the correct data and metadat from the columns of the
        # pandas.Dataframe `df_sel` which holds all entries of one CML ID.
        #
        # To understand this code, one has to be familiar with the concept
        # of xarray.Dataset and how it assigns dimensions and coordinates.
        # Please consult the xarray documentation.
        ds_list.append(
            xr.Dataset(
                data_vars=dict(
                    pmin=("time", df_sel.pmin.values), pmax=("time", df_sel.pmax.values)
                ),
                coords=dict(
                    time=df_sel.index.values,
                    cml_id=df_sel.id.values[0],
                    length=df_sel.pathlength.values[0],
                    frequency=df_sel.frequency.values[0],
                    site_a_longitude=df_sel.xstart.values[0],
                    site_b_longitude=df_sel.xend.values[0],
                    site_a_latitude=df_sel.ystart.values[0],
                    site_b_latitude=df_sel.yend.values[0],
                ),
            )
        )

    # concat all individual xarray datasets
    ds2012 = xr.concat(ds_list, dim="cml_id")

    return ds2012
