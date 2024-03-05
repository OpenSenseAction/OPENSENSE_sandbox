#  Personal Weather Station (PWS) quality control using python.
 -----------------------------------------------------------------------------------------------
### **How to cite:**

https://doi.org/10.5281/zenodo.4501919

### Reference paper:
Bárdossy, A., Seidel, J., and El Hachem, A.: The use of personal weather station observations to improve precipitation estimation and interpolation, Hydrol. Earth Syst. Sci., 25, 583–601, https://doi.org/10.5194/hess-25-583-2021, 2021.

-----------------------------------------------------------------------------------------------
### Main Procedure


**Flowchart from raw PWS data to filtered data for interpolation**

The three main codes corresponding to the IBF, Bias correction and EBF are available in the python_code folder

![flowchart_netatmo_paper](https://user-images.githubusercontent.com/22959071/106765543-3303fb00-6639-11eb-92d8-d0e06a6044f1.png)


-----------------------------------------------------------------------------------------------
### Indicator based Filter IBF


**Corresponsing code**
_02_pws_indicator_correlation_IBF.py

****Required Input****
  1. Hdf5 data file containing the PWS station data, the corresponding timestamps and 
    their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps and
    their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe containing mainly the correlation values between each PWS and the corresponding neighboring station data
    and the correlation between the neighboring primary network stations .
  2. The final result is obtained by keeping all the PWS where the correlation between PWS and 
    primary network is greater or equal to the correlation between the primary network stations.

**Results**: indicator filter applied to the test_data

![indic_corr_99](https://user-images.githubusercontent.com/22959071/106903818-c5200800-66fa-11eb-9efc-8e21011791c5.png)

**Note**: the present code is slighty different than the one in the original paper. A similar code will be uploaded soon.

-----------------------------------------------------------------------------------------------
### Bias correction


**Corresponsing code**
_02_pws_bias_correction_BC.py

****Required Input****
  1. Hdf5 data file containing the ***filtered*** PWS station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe for each PWS with the new data, a 'complete' new timeseries, used later on (for example in the interpolation)
 
**Results**: example of corrected data of one PWS
![pws_stn_70_ee_50_00_0b_40](https://user-images.githubusercontent.com/22959071/106904335-5d1df180-66fb-11eb-8937-8aaa24c43579.png)

-----------------------------------------------------------------------------------------------
### Event based filter (EBF)

**Corresponsing code**
_04_event_based_filter_EBF.py

****Required Input****
  1. Hdf5 data file containing the ***filtered and bias corrected*** PWS station data,
    the corresponding timestamps and their corresponding coordinates, in a metric coordinate system
  2. Hdf5 data file containing the primary network station data, the corresponding timestamps
    and their corresponding coordinates, in a metric coordinate system (same as PWS)
  
****Output****
  1. A dataframe containing for every event (or timestamp) the PWS that should be flagged and
    not used for the interpolation of the corresponding event or timestep
 
**Results** PWS stations that do not fit in the spatial neighboring are flagged

![event_date_2019_10_01 15_00_00](https://user-images.githubusercontent.com/22959071/106916083-13d39f00-6707-11eb-9d9a-7f3e76367063.png)
-----------------------------------------------------------------------------------------------

### Credit:
The precipitation data was downloaded from the German Weather Service (DWD) open data server which can be found under the following link: https://opendata.dwd.de/climate_environment/CDC/

The PWS data were downloaded using the Netatmo API: https://dev.netatmo.com/

### Note:
In the bias correction and event based filter an Ordinary Kriging implementation is used. This has not been yet uploaded but could be easily subtituted by the PyKrige code (10.5281/zenodo.3738604).
