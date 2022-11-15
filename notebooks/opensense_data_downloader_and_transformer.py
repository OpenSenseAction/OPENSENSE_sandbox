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
                'tl': (('sublink_id', 'time'), [df_data.s73.values, df_data.s83.values]),
                'frequency': (('sublink_id'), [73.5, 83.5]),
            },
            coords=dict(
                time=df_data.index.values,
                cml_id='cz_example_cml_1',
                length=4.866,
                sublink_id=(('sublink_id'), ['ab', 'ba']),
                site_0_lon=14.53,
                site_1_lon=14.53, # coordinates have been rounded for publication...
                site_0_lat=50.03,
                site_1_lat=50.03,
            ),
        )
        
    # add standard attributes
    ds = add_cml_attributes(ds)
        
                
    return ds
      

    
# 6 Eband CMLs from Czech Republic

download_fencl_2020_Eband_data = partial(
    download_data_file,
    url="https://zenodo.org/record/4090953/files/dataset.zip",
)

def transform_fencl_2020_Eband_data(local_path, fn):
    ds_list = []
    with zipfile.ZipFile(local_path + fn) as zfile:
        # read metadata
        df_metadata = pd.read_csv(
            zfile.open('raw/commercial_microwave_links_total_loss/metadata_table_commercial_microwave_links.csv'),
            index_col=0,
            sep=';',
        )
        
        for i, row in df_metadata.iterrows():
            for ab in ['a', 'b']:
                raw_data_fn = os.path.join('raw/commercial_microwave_links_total_loss', f'{i}{ab}.csv')
                print(f'Parsing raw data from {raw_data_fn}')
                # get file handle for CSV files stored in the ZIP file    
                f = zfile.open(raw_data_fn)
                df_data = pd.read_csv(
                    f,
                    index_col=0,
                    parse_dates=True,
                    sep=';',
                )
                
                ds_list.append(
                    xr.Dataset(
                        data_vars={'tl': (('time'), df_data.total_loss)},
                        coords=dict(
                            time=df_data.index.values,
                            cml_id=row.id_old,
                            length=row.length/1e3,
                            frequency=row[f'freq{ab.upper()}'],
                            site_0_lon=row.lonA,
                            site_1_lon=row.lonB,
                            site_0_lat=row.latA,
                            site_1_lat=row.latB,
                        ),
                    )
                )

    return ds_list


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
                    site_0_lon=df_sel.xstart.values[0],
                    site_1_lon=df_sel.xend.values[0],
                    site_0_lat=df_sel.ystart.values[0],
                    site_1_lat=df_sel.yend.values[0],
                ),
            )
        )

    # concat all individual xarray datasets
    ds2012 = xr.concat(ds_list, dim="cml_id")
    
    # add standard attributes
    ds2012 = add_cml_attributes(ds2012)

    return ds2012


# OpenMRG dataset from SMHI

download_andersson_2022_OpenMRG = partial(
    download_data_file,
    url="https://zenodo.org/record/7107689/files/OpenMRG.zip",
)

def transform_andersson_2022_OpenMRG(fn, path_to_extract_to):
    # For this ZIP file we cannot extract only the CML data since
    # the NetCDF with the CML data is quite large. This seems to
    # lead to crashes when reding directly from the ZIP file via Python.
    with zipfile.ZipFile(fn) as zfile:
        zfile.extractall(path_to_extract_to)
    
    # Read metadata and data
    df_metadata = pd.read_csv(os.path.join(path_to_extract_to, 'cml/cml_metadata.csv'), index_col=0)
    ds = xr.open_dataset(os.path.join(path_to_extract_to, 'cml/cml.nc'))
    
    # Add metadata with naming convention as currently used in pycomlink example data file
    for col_name, ds_var_name in [
        ('NearLatitude_DecDeg', 'site_0_lat'),
        ('NearLongitude_DecDeg', 'site_0_lon'),
        ('FarLatitude_DecDeg', 'site_1_lat'),
        ('FarLongitude_DecDeg', 'site_1_lon'),
        ('Frequency_GHz', 'frequency'),
        ('Polarization', 'polarization'),
        ('Length_km', 'length'),
    ]:
        ds.coords[ds_var_name] = (
            ('sublink'), 
            [df_metadata[df_metadata.Sublink==sublink_id][col_name].values[0] for sublink_id in list(ds.sublink.values)]
        )
        
    ds.attrs['comment'] += '\nMetadata added with preliminary code from opensense_data_downloader.py'
    
    # add standard attributes
    ds = add_cml_attributes(ds)
    
    return ds


def transform_German_CML_data(fn):
    
    ds = xr.open_dataset(fn)
    
    # rename according to new conventions
    ds = ds.rename({
        "site_a_latitude": "site_0_lat",
        "site_a_longitude": "site_0_lon",
        "site_b_latitude": "site_1_lat",
        "site_b_longitude": "site_1_lon",
        "channel_id": "sublink_id",
    })
    
    # add standard attributes
    ds = add_cml_attributes(ds)
    
    return ds


def add_cml_attributes(ds):
    
    # dictionary of optional and required attributes for variables
    # according to white paper draft
    dict_attributes = {
        "time": {
            "units": "s",
            "long_name": "time_utc"
        },
        "cml_id": {
            "long_name": "commercial_microwave_link_identifier"
        },
        'sublink_id': {
            "long_name": "sub-link_identifier"
        },
        'site_0_lat': {
            "units": "degrees in WGS84 projection",
            "long_name": "site_0_latitude",
        },
        'site_0_lon': {
            "units": "degrees in WGS84 projection",
            "long_name": "site_0_longitude",
        },      
        'site_0_elevation': {
            "units": "meters_above_sea",
            "long_name": "ground_elevation_above_sea_level",
        },     
        'site_0_altitude': {
            "units": "meters_above_sea",
            "long_name": "antenna_altitude_above_sea_level",
        }, 
        'site_1_lat': {
            "units": "degrees in WGS84 projection",
            "long_name": "site_1_latitude",
        },
        'site_1_lon': {
            "units": "degrees in WGS84 projection",
            "long_name": "site_1_longitude",
        },               
        'site_1_elevation': {
            "units": "meters_above_sea",
            "long_name": "ground_elevation_above_sea_level",
        },     
        'site_1_altitude': {
            "units": "meters_above_sea",
            "long_name": "antenna_altitude_above_sea_level",
        },    
        'length': {
            "units": "m",
            "long_name": "distance_between_pair_of_antennas",
        },                
        'frequency': {
            "units": "MHz",
            "long_name": "sublink_frequency",
        },         
        'tsl': {
            "units": "dBm",
            "long_name": "transmitted_signal_level",
        },             
        'rsl': {
            "units": "dBm",
            "long_name": "received_signal_level",
        },
        'polarization': {
            "units": "no units",
            "long_name": "sublink_polarization",
        }
    }
    
    # list of global attributes according to white paper draft
    global_attr_vars = [
        "title",
        "file author/s",
        "institution",
        "date",
        "source",
        "history",
        "naming convention",
        "license restrictions",
        "reference",
        "comment",
    ]
    
    # extract list of variables present in dataset
    ds_vars = list(ds.coords) + list(ds.data_vars)

    # add attributes of variables to dataset
    for v in ds_vars:
        if v in dict_attributes.keys():
            ds[v].attrs = dict_attributes[v]
        
    # add a placeholder for global attributes that are not given
    for v in global_attr_vars:
        if v not in ds.attrs.keys():            
            ds.attrs[v] = "NA"
            
    # set encoding attributes
    ds.time.encoding['units'] = "seconds since 1970-01-01 00:00:00"

    return ds

        
def check_existence_of_required_vars(ds):
        
    required_vars = [
        "time",
        "cml_id",
        "sublink_id",
        "site_0_lat",
        "site_0_lon",
        "site_1_lat",
        "site_1_lon",
        "frequency",
        "tsl",
        "rsl",
    ]  
    ds_vars = list(ds.coords) + list(ds.data_vars)    

    for required_var in required_vars:
        if required_var not in ds_vars:
            print("Warning: %s is required but not present."%required_var)
