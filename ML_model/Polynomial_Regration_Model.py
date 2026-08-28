"""
Code: for Polynomial Regression Model to compute wind farm power output
        and compare with Linear Regression.

# Link to wind turbine data source:
    # https://www.nsenergybusiness.com/projects/reusenkoge-wind-farm-expansion/?cf-view
    
# Power Curve Data Source:
    # https://en.wind-turbine-models.com/turbines/693-vestas-v112-3.3

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class WindFarmModel:
    
    # -------------------------------------------------------
    # Step 1: INITIALIZATION, STORE DATA AND TURBINE INFO
    # -------------------------------------------------------

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
    # Step 2: COMPUTE HOURLY POWER USING POWER CURVE
    # -------------------------------------------------------

    # Hourly Based
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
    # Step 3: Merge to DAILY and MONTHLY POWER
    # -------------------------------------------------------
    def aggregate_daily(self):
        self.df["day"] = self.df["valid_time"].dt.date
        daily_power = self.df.groupby("day")["farm_power_MW"].sum().reset_index()
        self.daily_power = daily_power
        return daily_power
    
    def aggregate_monthly(self):
        self.df["month"] = self.df["valid_time"].dt.month
        monthly_power = self.df.groupby("month")["farm_power_MW"].sum().reset_index()
        self.monthly_power = monthly_power
        return monthly_power

    # -------------------------------------------------------
    # Step 4: POLYNOMIAL REGRESSION PREDICTION
    # -------------------------------------------------------

    # Hourly Based
    def fit_polynomial_trend(self, degree: int = 3):
        X = self.monthly_power["month"].values.reshape(-1, 1)
        y = self.monthly_power["farm_power_MW"].values

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        self.y_pred = model.predict(X_poly)
        return self.y_pred
    
    # Daily Based
    def fit_polynomial_trend_daily(self, degree=3):
        X = np.arange(len(self.daily_power)).reshape(-1, 1)
        y = self.daily_power["farm_power_MW"].values

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        self.daily_poly_pred = model.predict(X_poly)
        return self.daily_poly_pred

    # ------------------------------------------------------
    # Step 5: Linear Regression Prediction
    # -----------------------------------------------------

    # Hourly Based
    def fit_linear_regression(self):
        """Fit simple linear regression to monthly production."""
        X = self.monthly_power["month"].values.reshape(-1, 1)
        y = self.monthly_power["farm_power_MW"].values

        model = LinearRegression()
        model.fit(X, y)

        self.linear_model = model
        self.linear_pred = model.predict(X)

        return self.linear_pred
    
    # Daily Based
    def fit_linear_regression_daily(self):
        X = np.arange(len(self.daily_power)).reshape(-1, 1)
        y = self.daily_power["farm_power_MW"].values

        model = LinearRegression()
        model.fit(X, y)

        self.daily_lin_pred = model.predict(X)
        return self.daily_lin_pred

    
    # -------------------------------------------------------
    # Step 6:EVALUATE MODELS
    # -------------------------------------------------------

    def evaluate_models(self):
        """Compare Linear vs Polynomial regression using standard metrics."""

        y_true = self.monthly_power["farm_power_MW"].values
        # y_true = self.daily_power["farm_power_MW"].values

        # Polynomial predictions
        y_poly = self.y_pred
        # y_poly = self.daily_poly_pred

        # Linear predictions
        y_lin = self.linear_pred
        # y_lin = self.daily_lin_pred

        # Compute metrics
        results = {
            "Polynomial Regression": {
                "R2": r2_score(y_true, y_poly),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_poly)),
                "MAE": mean_absolute_error(y_true, y_poly)
            },
            "Linear Regression": {
                "R2": r2_score(y_true, y_lin),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_lin)),
                "MAE": mean_absolute_error(y_true, y_lin)
            }
        }

        return results

    # -------------------------------------------------------
    # Step 7:PLOT HOURLY PRODUCTION
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

        # Linear regression trend on hourly data 
        lin_model = LinearRegression() 
        lin_model.fit(months, farm_power) 
        trend_linear = lin_model.predict(months)

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

        # Linear regression trend 
        fig.add_trace(go.Scatter( x=time, y=trend_linear, 
                                 mode='lines', name='Linear Regression Trend', 
                                 line=dict(color='green', width=2, dash='dash') ))


        fig.update_layout(
            title=f'Hourly Wind Farm Power Production with ML',
            xaxis=dict(title='Time', rangeslider=dict(visible=True)),
            yaxis=dict(title='Power [MW]'),
            template='plotly_white'
        )

        fig.write_html(filename)
        print(f"Interactive plot saved to {filename}")

    # -------------------------------------------------------
    # Step 8:PLOT DAILY PRODUCTION
    # -------------------------------------------------------
    
    # def plot_daily_production(self, filename):
    #     days = self.daily_power["day"]
    #     actual = self.daily_power["farm_power_MW"]

    #     fig = go.Figure()

    #     fig.add_trace(go.Scatter(
    #         x=days, y=actual,
    #         mode="lines",
    #         name="Daily Power [MW]",
    #         line=dict(color="blue")
    #     ))

    #     fig.add_trace(go.Scatter(
    #         x=days, y=self.daily_poly_pred,
    #         mode="lines",
    #         name="Polynomial Trend",
    #         line=dict(color="red")
    #     ))

    #     fig.add_trace(go.Scatter(
    #         x=days, y=self.daily_lin_pred,
    #         mode="lines",
    #         name="Linear Trend",
    #         line=dict(color="green", dash="dash")
    #     ))

    #     fig.update_layout(
    #         title="Daily Wind Farm Power Production with ML Trends",
    #         xaxis_title="Day",
    #         yaxis_title="Power [MW]",
    #         template="plotly_white"
    #     )

    #     fig.write_html(filename)
    #     print(f"Daily plot saved to {filename}")

