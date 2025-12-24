import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class WindFarmModel:
    """
    Compute hourly wind farm power using physics formula and fit polynomial regression.
    """

    def __init__(self, df: pd.DataFrame, n_turbines: int = 51, total_capacity: float = 175,
                 rotor_radius: float = 61, Cp: float = 0.45, air_density: float = 1.225):
        self.df = df.copy()
        self.n_turbines = n_turbines
        self.total_capacity = total_capacity
        self.rotor_radius = rotor_radius
        self.Cp = Cp
        self.rho = air_density
        self.model = None
        self.X_poly = None
        self.y_pred = None

    def compute_hourly_power(self, cut_in: float = 3, cut_out: float = 25):
        """Compute farm power using full physics formula."""
        A = np.pi * self.rotor_radius**2  # m²
        # Power per turbine in MW
        self.df['turbine_power_MW'] = 0.5 * self.rho * A * self.Cp * self.df['wind_speed']**3 / 1e6

        # Apply cut-in / cut-out
        self.df.loc[self.df['wind_speed'] < cut_in, 'turbine_power_MW'] = 0
        self.df.loc[self.df['wind_speed'] > cut_out, 'turbine_power_MW'] = 0

        # Total farm power
        self.df['farm_power_MW'] = self.df['turbine_power_MW'] * self.n_turbines
        self.df['farm_power_MW'] = self.df['farm_power_MW'].clip(upper=self.total_capacity)
        return self.df

    def aggregate_monthly(self):
        """Aggregate hourly farm power to monthly totals."""
        self.df['month'] = self.df['valid_time'].dt.month
        monthly_power = self.df.groupby('month')['farm_power_MW'].sum().reset_index()
        self.monthly_power = monthly_power
        return monthly_power

    def fit_polynomial_trend(self, degree: int = 3):
        """Fit polynomial regression to monthly production."""
        X = self.monthly_power['month'].values.reshape(-1,1)
        y = self.monthly_power['farm_power_MW'].values
        poly = PolynomialFeatures(degree=degree)
        self.X_poly = poly.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(self.X_poly, y)
        self.y_pred = self.model.predict(self.X_poly)
        return self.y_pred

    

    def plot_hourly_production_html(self, filename: str):
        """
        Plot hourly farm power with polynomial trend and save as interactive HTML.
        """
        # Hourly data
        time = self.df['valid_time']
        farm_power = self.df['farm_power_MW']

        # Polynomial trend over months (for display, repeated hourly)
        months = self.df['month'].values.reshape(-1,1)
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(months)
        model = LinearRegression()
        model.fit(X_poly, farm_power)
        trend_power = model.predict(X_poly)

        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=farm_power,
            mode='lines+markers',
            name='Hourly Farm Power [MW]',
            line=dict(color='skyblue'),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=time, y=trend_power,
            mode='lines',
            name='Polynomial Trend',
            line=dict(color='red', width=2)
        ))

        # Layout
        fig.update_layout(
            title=f'Hourly Wind Farm Power Production (Total Capacity {self.total_capacity} MW)',
            xaxis=dict(title='Time', rangeslider=dict(visible=True)),
            yaxis=dict(title='Power [MW]'),
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=1, y=1, xanchor='right', yanchor='top')
        )

        # Save interactive HTML
        fig.write_html(filename)
        print(f'Interactive plot saved to {filename}')

