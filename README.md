# Wind-to-Hydrogen Arbitrage Model 

## Project Overview
This project models the techno-economic viability of co-locating a **50 MW PEM Electrolyzer** with the **300 MW Bürgerwindpark Reußenköge** onshore wind farm in Northern Germany.

Using a Python-based dispatch model, the system performs hourly arbitrage between:
1.  **Selling electricity to the grid** (Day-Ahead Market).
2.  **Producing Green Hydrogen** when prices are low or negative.

## Key Results 
* **Revenue Uplift:** €21.2 Million / year compared to the wind-only baseline.
* **Payback Period:** 4.2 Years.
* **LCOH:** €5.06 / kg (Competitive with market rates).
* **Conclusion:** The arbitrage strategy successfully mitigates the risk of negative prices while creating a new revenue stream.

![Payback Heatmap](images/payback_heatmap.png)
*(Sensitivity Analysis of Payback Period vs. H2 Price and CAPEX)*

## Data Sources 
* **Wind Data:** ERA5 Reanalysis (Copernicus CDS) for location 54.60°N, 8.90°E.
* **Price Data:** ENTSO-E Day-Ahead Prices for DE-LU (2024).
* **Turbine:** Vestas V150-5.6 MW Power Curve.
* **Electrolyzer:** Siemens Silyzer 300 specs (52.2 kWh/kg efficiency).

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Download the `.nc` wind data from Copernicus (see instructions in notebook).
3.  Run `main_model.ipynb`.

## Author

Master's Student in Wind Energy Engineering
