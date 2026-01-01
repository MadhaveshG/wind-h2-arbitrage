import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

@dataclass
class ComparisonParams:
    h2_price_eur_per_kg: float = 4.0
    var_om_eur_per_kg: float = 0.20     # Variable Operation & Maintenance cost per kilogram of hydrogen produced.
    specific_kwh_per_kg: float = 52.2   # The amount of electricity (in kWh) required to produce 1 kg of hydrogen in the electrolyzer
    electrolyzer_cap_mw: float = 50.0   # The maximum power capacity of the electrolyzer in megawatts (MW).
    min_load_fraction: float = 0.10     # 10% => 5 MW
    strict_decision: bool = False       # if True, follow store_decision strictly (ignore p*)
    allow_grid_import: bool = False     # minimal setup: keep False


def _compute_hybrid_row(wind_mw, price, decision, params: ComparisonParams, p_star: float):
    # default outputs
    electrolyzer_mw = 0.0
    sell_mw = wind_mw
    h2_kg = 0.0

    # decision rule
    condition = decision == 1 if params.strict_decision else (decision == 1 and price <= p_star)

    if condition:
        el_from_wind = min(wind_mw, params.electrolyzer_cap_mw)
        min_load_mw = params.electrolyzer_cap_mw * params.min_load_fraction

        if el_from_wind < min_load_mw and not params.allow_grid_import:
            # below min-load; electrolyzer off in minimal setup
            electrolyzer_mw = 0.0
            sell_mw = wind_mw
            h2_kg = 0.0
        else:
            # If allow_grid_import=True, you can add grid top-up logic here later
            electrolyzer_mw = el_from_wind
            sell_mw = max(0.0, wind_mw - electrolyzer_mw)
            h2_kg = (electrolyzer_mw * 1000.0) / params.specific_kwh_per_kg

    # revenues & costs (per hour)
    rev_elec = sell_mw * price
    rev_h2 = h2_kg * params.h2_price_eur_per_kg
    var_om = h2_kg * params.var_om_eur_per_kg
    grid_cost = 0.0  # no grid import in minimal version

    hybrid_net = rev_elec + rev_h2 - var_om - grid_cost
    return electrolyzer_mw, sell_mw, h2_kg, rev_elec, rev_h2, var_om, grid_cost, hybrid_net


def generate_interactive_comparison(
    final_csv_path: str,
    graphs_dir: str,
    params: ComparisonParams = ComparisonParams()
):
    """
    Create interactive comparison graphs (Sell-only vs Hybrid) from final arbitrage CSV.

    CSV must have columns: date,hour,farm_power_MW,price_EUR_MWh,store_decision
    Outputs saved in graphs_dir (HTML if Plotly is available; otherwise PNG).
    """
    os.makedirs(graphs_dir, exist_ok=True)

    # Load CSV
    if not os.path.exists(final_csv_path):
        raise FileNotFoundError(f"File not found: {final_csv_path}")

    df = pd.read_csv(final_csv_path)
    expected = {"date", "hour", "farm_power_MW", "price_EUR_MWh", "store_decision"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected: {expected}")

    # Datetime
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("datetime")

    # Standardize names
    df = df.rename(columns={
        "farm_power_MW": "wind_power_MW",
        "price_EUR_MWh": "price_EUR_per_MWh"
    })

    # Sell-only baseline
    df["sell_only_hourly_net_eur"] = df["wind_power_MW"] * df["price_EUR_per_MWh"]

    # Breakeven p* [€/MWh] = ((H2_price - var_OM)/kWh/kg) * 1000
    p_star = ((params.h2_price_eur_per_kg - params.var_om_eur_per_kg) / params.specific_kwh_per_kg) * 1000.0

    # Hybrid dispatch
    results = df.apply(
        lambda r: _compute_hybrid_row(
            r["wind_power_MW"], r["price_EUR_per_MWh"], r["store_decision"], params, p_star
        ), axis=1
    )
    df[["electrolyzer_mw", "electricity_sold_mw", "h2_produced_kg",
        "revenue_electricity_eur", "revenue_h2_eur",
        "var_om_cost_eur", "cost_electricity_import_eur",
        "hybrid_hourly_net_eur"]] = pd.DataFrame(results.tolist(), index=df.index)

    df["p_star_eur_per_mwh"] = p_star
    df["delta_hybrid_minus_sell_only_eur"] = df["hybrid_hourly_net_eur"] - df["sell_only_hourly_net_eur"]

    # Totals (console only)
    total_sell = df["sell_only_hourly_net_eur"].sum()
    total_hybrid = df["hybrid_hourly_net_eur"].sum()
    total_delta = df["delta_hybrid_minus_sell_only_eur"].sum()

    print("\n=== INTERACTIVE GRAPH COMPARISON SUMMARY ===")
    print(f"H2 price (€/kg): {params.h2_price_eur_per_kg}")
    print(f"Breakeven p* (€/MWh): {p_star:.2f}")
    print(f"Total Sell-only net (EUR): {total_sell:,.2f}")
    print(f"Total Hybrid net (EUR): {total_hybrid:,.2f}")
    print(f"Total Δ (Hybrid − Sell-only) (EUR): {total_delta:,.2f}")

    # --------- PLOTS ----------
    if PLOTLY_AVAILABLE:
        # 1) Hourly net comparison
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["datetime"], y=df["sell_only_hourly_net_eur"], mode="lines",
                                  name="Sell-only hourly net (EUR)", line=dict(color="#1f77b4")))
        fig1.add_trace(go.Scatter(x=df["datetime"], y=df["hybrid_hourly_net_eur"], mode="lines",
                                  name="Hybrid hourly net (EUR)", line=dict(color="#ff7f0e")))
        fig1.update_layout(title="Hourly Net Value: Sell-only vs Hybrid (Store as H₂)",
                           xaxis_title="Time", yaxis_title="EUR per hour",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                           template="plotly_white")
        fig1.write_html(os.path.join(graphs_dir, "interactive_hourly_net_comparison.html"), include_plotlyjs="cdn")

        # 2) Cumulative net
        df["sell_cum"] = df["sell_only_hourly_net_eur"].cumsum()
        df["hyb_cum"] = df["hybrid_hourly_net_eur"].cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["datetime"], y=df["sell_cum"], mode="lines",
                                  name="Sell-only cumulative (EUR)", line=dict(color="#1f77b4")))
        fig2.add_trace(go.Scatter(x=df["datetime"], y=df["hyb_cum"], mode="lines",
                                  name="Hybrid cumulative (EUR)", line=dict(color="#ff7f0e")))
        fig2.update_layout(title="Cumulative Net Value Over Time",
                           xaxis_title="Time", yaxis_title="Cumulative EUR",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                           template="plotly_white")
        fig2.write_html(os.path.join(graphs_dir, "interactive_cumulative_net_comparison.html"), include_plotlyjs="cdn")

        # 3) Revenue & cost composition (sampled bars)
        N = max(1, int(len(df) / 1000))
        sample = df.iloc[::N, :].copy()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=sample["datetime"], y=sample["revenue_electricity_eur"], name="Electricity revenue", marker_color="#1f77b4"))
        fig3.add_trace(go.Bar(x=sample["datetime"], y=sample["revenue_h2_eur"], name="H₂ revenue", marker_color="#2ca02c"))
        fig3.add_trace(go.Bar(x=sample["datetime"], y=-sample["var_om_cost_eur"], name="Variable O&M (cost)", marker_color="#d62728"))
        fig3.update_layout(barmode="relative", title="Hybrid: Revenue & Variable Cost Composition (sampled hours)",
                           xaxis_title="Time", yaxis_title="EUR per hour",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                           template="plotly_white")
        fig3.write_html(os.path.join(graphs_dir, "interactive_hybrid_revenue_cost_stack.html"), include_plotlyjs="cdn")

        # 4) Electrolyzer operation over time
        df["day"] = df["date"].dt.date
        pivot = df.pivot_table(index="hour", columns="day", values="store_decision", aggfunc="mean", fill_value=0)
        fig4 = px.imshow(pivot.values, aspect="auto", color_continuous_scale=["#d62728", "#2ca02c"],
                         labels=dict(color="Store decision (mean)"),
                         x=pivot.columns.astype(str), y=pivot.index,
                         title="Store Decision Heatmap (Hour vs Day)")
        fig4.update_layout(xaxis_title="Day", yaxis_title="Hour", template="plotly_white")
        fig4.write_html(os.path.join(graphs_dir, "interactive_store_decision_heatmap.html"), include_plotlyjs="cdn")

        print("\nSaved interactive figures in:", graphs_dir)

    else:
        # Fallback to static PNGs if Plotly not available
        import matplotlib.pyplot as plt

        # 1) Hourly net comparison
        plt.figure(figsize=(12, 5))
        plt.plot(df["datetime"], df["sell_only_hourly_net_eur"], label="Sell-only hourly net (EUR)", color="#1f77b4", linewidth=1.3)
        plt.plot(df["datetime"], df["hybrid_hourly_net_eur"], label="Hybrid hourly net (EUR)", color="#ff7f0e", linewidth=1.3)
        plt.title("Hourly Net Value: Sell-only vs Hybrid (Store as H₂)")
        plt.xlabel("Time"); plt.ylabel("EUR per hour"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(graphs_dir, "plot_hourly_net_comparison.png"), dpi=150); plt.close()

        # 2) Cumulative net
        plt.figure(figsize=(12, 5))
        plt.plot(df["datetime"], df["sell_only_hourly_net_eur"].cumsum(), label="Sell-only cumulative (EUR)", color="#1f77b4", linewidth=1.5)
        plt.plot(df["datetime"], df["hybrid_hourly_net_eur"].cumsum(), label="Hybrid cumulative (EUR)", color="#ff7f0e", linewidth=1.5)
        plt.title("Cumulative Net Value Over Time")
        plt.xlabel("Time"); plt.ylabel("Cumulative EUR"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(graphs_dir, "plot_cumulative_net_comparison.png"), dpi=150); plt.close()

        # 4) Revenue & cost composition (sampled)
        N = max(1, int(len(df) / 2000))
        sample = df.iloc[::N, :]
        x = np.arange(len(sample))
