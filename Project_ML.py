"""
Profitability Prediction Using Machine Learning for Onshore Wind Farms

Here, We have used two ML models 
1. Polynomial Regression
2. Logistic Regression

Created on: 24/12/2025
Created by: M. Eng. Karan Soni, Madhvesh Gorakhiya
Supervisor: Prof. Dr. Andreas Heinen

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr

from ML_model.Polynomial_Regration_Model import WindFarmModel

"""
 Step 1: 
    - Load the Wind Data
    - Finding Wind Speed at 100m height 
    - Save as CSV file

"""
# Load the NetCDF file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
Wind_Data = os.path.join(BASE_DIR, "DATA", "Wind_data_ERA5", "Reusenkoge_onshore_wind_data.nc") 
ds = xr.open_dataset(Wind_Data) # Load the NetCDF file using xarray

# Finding the wind speed at 100m height
ds['wind_speed'] = np.sqrt(ds['u100']**2 + ds['v100']**2)
ds = ds.mean(dim=['latitude', 'longitude'])  # Average over spatial dimensions
df = ds[['wind_speed']].to_dataframe().reset_index()
print(df.head())
df = df.dropna()
df['hour'] = df['valid_time'].dt.hour
df['month'] = df['valid_time'].dt.month

# Save the CSV File of Valid time Vs Wind Speed at 100m height
processed_data_path = os.path.join(BASE_DIR, "DATA", "Wind_data_ERA5", "wind_speed.csv")
df.to_csv(processed_data_path, index=False)

"""
Step 2: Compute farm power & aggregate monthly
"""

wind_farm = WindFarmModel(df, n_turbines=51, total_capacity=175,
                          rotor_radius=61, Cp=0.45, air_density=1.225)
wind_farm.compute_hourly_power(),
wind_farm.aggregate_monthly()

"""
Step 3: Fit polynomial regression
"""
wind_farm.fit_polynomial_trend(degree=3)

"""
Step 4: Plot monthly production with trend
"""

wind_farm.plot_hourly_production_html(
    filename=os.path.join(BASE_DIR, "DATA", "Wind_data_ERA5", "hourly_power_plot.html")
)


