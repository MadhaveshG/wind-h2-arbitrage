## 📖 What is this project?
This project is a computer simulation for a real wind farm in Northern Germany (**Bürgerwindpark Reußenköge**). 

It answers a big question: **"Can we make more money by turning wind energy into hydrogen instead of just selling it to the grid?"**

I modeled a system that connects the **210 MW wind farm** to a **50 MW Hydrogen Electrolyzer** to see if it makes financial sense.

## ❓ The Problem
Wind farms in Germany face two big issues:
1.  **Low Prices:** When it's very windy, electricity prices often drop to **zero** (or even negative).
2.  **Wasted Energy:** The grid gets "clogged" (congestion), forcing wind farms to turn off. This is called **curtailment**.

## 💡 The Solution
I wrote a Python program that acts like a "Smart Operator":
* **Step 1:** It watches the electricity price every hour.
* **Step 2:** If prices are **HIGH**, it sells electricity to the grid (Profit!).
* **Step 3:** If prices are **LOW**, it turns on the Electrolyzer to make Green Hydrogen (Profit!).
* **Step 4:** It also captures energy that the grid would normally throw away.

## 📊 Key Results
Here is what the model found for the year 2024:

| Metric | Wind Farm Only | Wind + Hydrogen (My Model) |
| :--- | :--- | :--- |
| **Total Revenue** | €[Insert 210MW Value] | **€61,006,897.12** |
| **Improvement** | - | **🟢 +[X]% Extra Profit** |
| **Hydrogen Made** | 0 tons | **6,629.44 tons** |
| **Energy Saved** | 0 GWh | **61.6 GWh** (Saved from waste) |

> **Bottom Line:** Adding hydrogen creates a **4.2-year payback period** and makes the wind farm much more profitable.

## 📂 Project Files
* `main_model.ipynb` → The main code (Start here!).
* `data/` → Folder containing wind and price data.
* `src/` → Helper scripts for the turbine and electrolyzer logic.
* `images/` → Graphs and charts.

## ⚙️ How to Run It
1.  **Install Python libraries:**
    ```bash
    pip install pandas numpy matplotlib xarray
    ```
2.  **Download Data:** (Instructions inside the notebook).
3.  **Run the Notebook:** Open `main_model.ipynb` and click "Run All".

## 👤 Author
**Madhvesh & Karan**
*Master's Student in Wind Energy Engineering*
