"""
Code for Logistic Regression Model to decide when to STORE or SELL energy
Output: There two main outputs:
    1. Value 0 : SELL energy
    2. Value 1 : STORE energy 
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class LogisticArbitrageModel:

    def __init__(self, power_df: pd.DataFrame, price_df: pd.DataFrame):
        self.power_df = power_df.copy()
        self.price_df = price_df.copy()
        self.daily_data = None
        self.model = None

    # --------------------------------------------------
    # STEP 1: CREATE SEASONAL DEMAND INDEX
    # --------------------------------------------------

    @staticmethod
    def seasonal_demand_factor(month):
        if month in [12, 1, 2]:      # Winter
            return 1.2
        elif month in [3, 4, 5]:     # Spring
            return 0.9
        elif month in [6, 7, 8]:     # Summer
            return 0.8
        else:                        # Autumn
            return 1.0

    # --------------------------------------------------
    # STEP 2: PREPARE DATASET of POWER + PRICE
    # --------------------------------------------------

    def prepare_dataset(self):
        # Convert wind timestamps to UTC
        self.power_df["valid_time"] = pd.to_datetime(self.power_df["valid_time"], utc=True)

        # Convert price timestamps to UTC
        self.price_df["Date (GMT+1)"] = pd.to_datetime(self.price_df["Date (GMT+1)"], utc=True)
        self.price_df["Date (GMT+1)"] = self.price_df["Date (GMT+1)"].dt.tz_convert("UTC")

        # Strip column whitespace and rename price column
        self.price_df.columns = self.price_df.columns.str.strip()
        price_col = [c for c in self.price_df.columns if "Price" in c and "EUR" in c]
        if not price_col:
            raise ValueError("No price column found in price_df!")
        self.price_df = self.price_df.rename(columns={price_col[0]: "price_EUR_MWh"})

        # Merge power and price data
        df = pd.merge(
            self.power_df,
            self.price_df,
            left_on="valid_time",
            right_on="Date (GMT+1)",
            how="inner"
        )

        df = df.dropna(subset=["farm_power_MW", "price_EUR_MWh"])

        # # Daily aggregation
        # df["day"] = df["valid_time"].dt.date
        # df["month"] = df["valid_time"].dt.month

        # daily = df.groupby("day").agg({
        #     "farm_power_MW": "sum",
        #     "price_EUR_MWh": "mean",
        #     "month": "first"
        # }).reset_index()

        # # Seasonal demand
        # daily["seasonal_demand"] = daily["month"].apply(self.seasonal_demand_factor)

        # # Decision: STORE if price < 25, else SELL
        # daily["store_decision"] = np.where(daily["price_EUR_MWh"] < 25, 1, 0)

        # self.daily_data = daily
        # return daily

        # Hourly-based
        df["month"] = df["valid_time"].dt.month
        df["seasonal_demand"] = df["month"].apply(self.seasonal_demand_factor)

        # Decision: STORE if price < 25, else SELL
        df["store_decision"] = np.where(df["price_EUR_MWh"] < 25, 1, 0)

        # Store hourly data
        self.hourly_data = df
        return df

    # --------------------------------------------------
    # STEP 3: TRAIN LOGISTIC REGRESSION
    # --------------------------------------------------
    def train_model(self):
        features = ["farm_power_MW", "price_EUR_MWh", "seasonal_demand"]
        # X = self.daily_data[features]
        # y = self.daily_data["store_decision"]
        X = self.hourly_data[features]
        y = self.hourly_data["store_decision"]


        # Check if both classes have at least 2 samples
        if y.nunique() < 2 or y.value_counts().min() < 2:
            # Not enough samples to stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )


        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("\nLogistic Regression Results:\n")
        print(classification_report(y_test, y_pred))

        return self.model

    # --------------------------------------------------
    # STEP 4: COEFFICIENT INTERPRETATION
    # --------------------------------------------------
    def print_coefficients(self):
        if self.model is None:
            print("Model not trained yet.")
            return

        print("\nModel Coefficients:\n")
        for name, coef in zip(
            ["Power", "Price", "Seasonal Demand"],
            self.model.coef_[0]
        ):
            print(f"{name:20s}: {coef:.2f}")

    # --------------------------------------------------
    # STEP 5: INTERACTIVE LINE PLOT OF STORE/SELL
    # --------------------------------------------------
    def plot_decision_html(self, filename):
        df = self.hourly_data.copy()

        # Color coding for STORE/SELL
        df["color"] = df["store_decision"].map({1: "green", 0: "red"})
        df["decision_text"] = df["store_decision"].map({1: "STORE", 0: "SELL"})

        fig = go.Figure()

        # ----------------------------------------------------
        # 1. Improved Electricity Price Curve (smoothed)
        # ----------------------------------------------------
        fig.add_trace(go.Scatter(
            x=df["valid_time"],
            y=df["price_EUR_MWh"].rolling(3, center=True).mean(),  # smooth price
            mode="lines",
            line=dict(color="orange", width=2),
            name="Electricity Price (€/MWh)",
            yaxis="y2"
        ))

        # ----------------------------------------------------
        # 2. STORE/SELL Markers
        # ----------------------------------------------------
        fig.add_trace(go.Scatter(
            x=df["valid_time"],
            y=df["store_decision"],
            mode="markers",
            marker=dict(size=7, color=df["color"]),
            name="STORE (1) / SELL (0)",
            text=df["decision_text"],
            hovertemplate="<b>%{text}</b><br>"
                        "Time: %{x}<br>"
                        "Price: %{y2} €/MWh<br>"
                        "<extra></extra>"
        ))

        # ----------------------------------------------------
        # 3. Threshold Line (25 €/MWh)
        # ----------------------------------------------------
        fig.add_hline(
            y=25,
            line=dict(color="black", width=1, dash="dash"),
            annotation_text="25 €/MWh Threshold",
            annotation_position="top left",
            yref="y2"
        )

        # ----------------------------------------------------
        # Layout Improvements
        # ----------------------------------------------------
        fig.update_layout(
            title="STORE vs SELL Decisions with Improved Price Curve",
            xaxis=dict(title="Time", rangeslider=dict(visible=True)),
            yaxis=dict(
                title="Decision (0 = SELL, 1 = STORE)",
                tickvals=[0, 1],
                ticktext=["SELL", "STORE"]
            ),
            yaxis2=dict(
                title="Electricity Price (€/MWh)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_white",
            hovermode="x unified",

            # ⭐ LEGEND ON RIGHT SIDE
            legend=dict(
                x=1.05,
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)"
            )
        )

        fig.write_html(filename)
        print(f"Saved → {filename}")

    # --------------------------------------------------
    # STEP 6: SAVE STORE HOURS AND DAYS TO CSV
    # --------------------------------------------------
    def save_store_info(self, hours_file="store_hours.csv", days_file="store_days.csv"):
        if not hasattr(self, "hourly_data"):
            print("Hourly data not prepared yet.")
            return

        # STORE hours
        store_hours = self.hourly_data[self.hourly_data["store_decision"] == 1][["valid_time", "price_EUR_MWh"]]
        store_hours.to_csv(hours_file, index=False)
        print(f"Saved STORE hours → {hours_file}")

        # STORE days
        store_days = pd.DataFrame({
            "store_day": self.hourly_data[self.hourly_data["store_decision"] == 1]["valid_time"].dt.date.unique()
        })
        store_days.to_csv(days_file, index=False)
        print(f"Saved STORE days → {days_file}")

    # --------------------------------------------------
    # STEP 7: SAVE FINAL COMBINED CSV (DATE, TIME, POWER, PRICE, DECISION)
    # --------------------------------------------------
    def save_final_csv(self, filename="final_energy_arbitrage.csv"):
        if not hasattr(self, "hourly_data"):
            print("Hourly data not prepared yet. Run prepare_dataset() first.")
            return

        df = self.hourly_data.copy()

        # Extract clean date and hour
        df["date"] = df["valid_time"].dt.date
        df["hour"] = df["valid_time"].dt.hour

        # Select final columns
        final_df = df[[
            "date",
            "hour",
            "farm_power_MW",
            "price_EUR_MWh",
            "store_decision"
        ]]

        # Save CSV
        final_df.to_csv(filename, index=False)
        print(f"Final CSV saved → {filename}")

        return final_df

