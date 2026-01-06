"""
Profitability Prediction Using Machine Learning for Onshore Wind Farms

Location: Reusenkoog, Germany

Here, We have used two ML models 
1. Polynomial Regression and Linear Regression
2. Logistic Regression

Created on: 24/12/2025
Created by: Karan Soni, Madhvesh Gorakhiya
Supervisor: Prof. Dr. Andreas Heinen

Research Question:
    1. Can machine learning models accurately predict wind farm power output and 
        identify the optimal hours to STORE or SELL energy?
    2. If low price energy is stored as hydrogen, can the hybrid system 
        remain economically profitable without subsidies?

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr

from ML_model.Polynomial_Regration_Model import WindFarmModel
from ML_model.Logistic_Arbitrage_model import LogisticArbitrageModel
from ML_model.H2_model import generate_interactive_comparison, ComparisonParams
"""
 Step 1: 
    - Load the Wind Data
    - Finding Wind Speed at 100m height 
    - Save as CSV file

File Location: DATA/Wind_data_ERA5/Reusenkoge_onshore_wind_data.nc
"""
# Load the NetCDF file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
Wind_Data = os.path.join(BASE_DIR, "DATA", "Wind_data_ERA5", "Reusenkoge_onshore_wind_data.nc") 
ds = xr.open_dataset(Wind_Data) # Load the NetCDF file using xarray

# Finding the wind speed at 100m height
ds['wind_speed'] = np.sqrt(ds['u100']**2 + ds['v100']**2)
ds = ds.mean(dim=['latitude', 'longitude'])
df = ds[['wind_speed']].to_dataframe().reset_index()
df = df.dropna()
df['hour'] = df['valid_time'].dt.hour
df['day'] = df['valid_time'].dt.date
df['month'] = df['valid_time'].dt.month

# Save the CSV File of Valid time Vs Wind Speed at 100m height
processed_data_path = os.path.join(BASE_DIR, "DATA", "Wind_data_ERA5", "wind_speed.csv")
df.to_csv(processed_data_path, index=False)

"""
Step 2: Load Market Price Data
File Location: DATA/sport_price_2024.csv
"""

price_df = pd.read_csv(
    os.path.join(BASE_DIR, "DATA", "sport_price_2024.csv"),
    usecols=["Date (GMT+1)", "Price_EUR/MW"]
)

"""
Step 3: Compute farm power output based on power curve
"""

wind_speeds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17, 18, 19, 20, 21, 22, 23, 24, 25])  # m/s
power_kW =    np.array([0, 0, 22, 134, 302, 552, 906, 1370, 1950, 2586, 3071, 3266, 3298, 3300, 3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0])  # kW


wind_farm = WindFarmModel(
    df=df,
    wind_speeds=wind_speeds,
    power_kW=power_kW,
    n_turbines=51
)

"""
Step 4: Compute Hourly, Daily, Monthly Power Output
"""

wind_farm.compute_hourly_power()
wind_farm.aggregate_daily()
wind_farm.aggregate_monthly()


"""
Step 5: Compute polynomial regression and linear regression models
"""
# Hourly based models
wind_farm.fit_polynomial_trend(degree=3)
wind_farm.fit_linear_regression()

## Daily based models
# wind_farm.fit_polynomial_trend_daily()
# wind_farm.fit_linear_regression_daily()

results = wind_farm.evaluate_models() 
print(results)

"""
Step 6: Compute Logistic Arbitrage Model
"""

arb_model = LogisticArbitrageModel(
    power_df=wind_farm.df,
    price_df=price_df
)

arb_model.prepare_dataset()
arb_model.train_model()
arb_model.print_coefficients()

"""
Step 7: Plot Graphs
"""

# Plot 1: Hours Vs Power Plot (Linear and Polynomial Regression)
wind_farm.plot_hourly_production_html(
    filename=os.path.join(BASE_DIR,"Graphs", "hourly_power_plot.html")
)

## Daily Vs Power Plot (Linear and Polynomial Regression)
# wind_farm.plot_daily_production(
#     filename=os.path.join(BASE_DIR, "Graphs", "daily_power_plot.html")
# )

# Plot 2: Store Sell Decision Plot
arb_model.plot_decision_html(
    os.path.join(BASE_DIR, "Graphs", "store_sell_decision.html")
)

"""
Step 8: Save STORE information to CSV
"""

arb_model.save_store_info(
    hours_file=os.path.join(BASE_DIR, "Graphs", "store_hours.csv"),
    days_file=os.path.join(BASE_DIR, "Graphs", "store_days.csv")
)

arb_model.save_final_csv(
    filename=os.path.join(BASE_DIR, "Graphs", 
                          "final_energy_arbitrage.csv")
)

graphs_dir = os.path.join(BASE_DIR, "Graphs")

# If your file name is final_arbitary.csv, set that here; otherwise use final_energy_arbitrage.csv
final_csv_path = os.path.join(graphs_dir, "final_energy_arbitrage.csv")  # or "final_arbitary.csv"

params = ComparisonParams(
    h2_price_eur_per_kg=5.0,     
    var_om_eur_per_kg=0.20,
    specific_kwh_per_kg=52.2,
    electrolyzer_cap_mw=50.0,
    min_load_fraction=0.10,
    strict_decision=False,       # set True to obey store_decision strictly (ignore p*)
    allow_grid_import=False      # keep False for minimal version
)

generate_interactive_comparison(final_csv_path, graphs_dir, params)

