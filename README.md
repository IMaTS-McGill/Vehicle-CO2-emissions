# Vehicle CO2 emissions
 
## 1. On-road experiments data

There are two folders for the raw and processed data:

1. **Raw Data:** Includes a sub-folder for each category of vehicle (e.g. Class I - ICEV, Class I - HEV, Class II, etc.). Under each category sub-folder, there are the raw data files related to a specific experiment. For example, we can get the raw OBD and PEMS data for *Chevrolet Malibu 2019 (October 2019)* experiment with the following path:

```
- Raw Dat
  - Class I - ICEV
    - Chevrolet Malibu 2019 (October 2019)
      - OBD
      - PEMS
```
    
2. **Processed Data:** Includes a sub-folder for each category of vehicle (e.g. Class I - ICEV, Class I - HEV, Class II, etc.). Under each category sub-folder, there is the processed data file related to a specific experiment. For example, we can get the processed data for *Chevrolet Malibu 2019 (October 2019)* experiment with the following path:

```
- Processed Data
  - Class I - ICEV
    - Chevrolet_Malibu_2019_(October_2019).csv
```

The document titled `Data Collection and Description.pdf` contains information about how data is collected, as well as descriptions of the original variables obtained using OBD and PEMS equipment.

## 2. Data processing codes

When a new on-road experiment is completed, upload the raw data files to **Raw Data** folder (corresponding vehicle class/category sub-folder). Use the following naming convention for the files:

`Make Model Year (Collection Month Year)`

The following Jupyter Notebooks are used for data processing based on vehicle type:

- Data preprocessing - ICEV.ipynb
- Data preprocessing - HEV.ipynb
- Data preprocessing - Class II.ipynb

Ensure the following files are also in the main folder:

- utils.py
- variable_name_conversion_obd.csv
- variable_name_conversion_pems.csv

When the processing scripts run successfully without any errors, the processed data file is saved in the **Processed Data** folder (corresponding vehicle class/category sub-folder).
