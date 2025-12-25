# Link to wind turbine data source:
    # https://www.nsenergybusiness.com/projects/reusenkoge-wind-farm-expansion/?cf-view
    
# Power Curve Data Source:
    # https://en.wind-turbine-models.com/turbines/693-vestas-v112-3.3

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

class WindFarmModel:
    """
    Compute hourly wind farm power using a POWER CURVE instead of physics formula.
    """

    def __init__(self, df: pd.DataFrame, 
                wind_speeds: np.ndarray,
                power_kW: np.ndarray,
                n_turbines: int = 51):
            
        self.df = df.copy()
        self.n_turbines = n_turbines

        # Store power curve arrays from user
        self.wind_speeds = np.array(wind_speeds)
        self.power_kW = np.array(power_kW)

        # Create interpolator
        self.power_interp = interp1d(
        self.wind_speeds,
        self.power_kW,
        kind="linear",
        bounds_error=False,
        fill_value=0
        )

    # -------------------------------------------------------
    # COMPUTE HOURLY POWER USING POWER CURVE
    # -------------------------------------------------------
    def compute_hourly_power(self):
        """Compute farm power using turbine power curve."""
        
        # Interpolate turbine power (kW)
        self.df["turbine_power_kW"] = self.power_interp(self.df["wind_speed"])

        # Convert to MW
        self.df["turbine_power_MW"] = self.df["turbine_power_kW"] / 1000.0

        # Total farm power
        self.df["farm_power_MW"] = self.df["turbine_power_MW"] * self.n_turbines

        return self.df

    # -------------------------------------------------------
    # MONTHLY AGGREGATION
    # -------------------------------------------------------
    def aggregate_monthly(self):
        self.df["month"] = self.df["valid_time"].dt.month
        monthly_power = self.df.groupby("month")["farm_power_MW"].sum().reset_index()
        self.monthly_power = monthly_power
        return monthly_power

    # -------------------------------------------------------
    # POLYNOMIAL TREND
    # -------------------------------------------------------
    def fit_polynomial_trend(self, degree: int = 3):
        X = self.monthly_power["month"].values.reshape(-1, 1)
        y = self.monthly_power["farm_power_MW"].values

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        self.y_pred = model.predict(X_poly)
        return self.y_pred

    # -------------------------------------------------------
    # PLOT HOURLY PRODUCTION
    # -------------------------------------------------------
    def plot_hourly_production_html(self, filename: str):
        time = self.df["valid_time"]
        farm_power = self.df["farm_power_MW"]

        # Fit polynomial trend on hourly data
        months = self.df["month"].values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(months)
        model = LinearRegression()
        model.fit(X_poly, farm_power)
        trend_power = model.predict(X_poly)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=farm_power,
            mode='lines',
            name='Hourly Farm Power [MW]',
            line=dict(color='skyblue')
        ))
        fig.add_trace(go.Scatter(
            x=time, y=trend_power,
            mode='lines',
            name='Polynomial Trend',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=f'Hourly Wind Farm Power Production Capacity MW',
            xaxis=dict(title='Time', rangeslider=dict(visible=True)),
            yaxis=dict(title='Power [MW]'),
            template='plotly_white'
        )

        fig.write_html(filename)
        print(f"Interactive plot saved to {filename}")
