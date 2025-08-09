
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant

st.set_page_config(page_title="Finans App – Afkast & Simulation", layout="wide")

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Serie"] = df["Serie"].astype(str)
    df["Produktudbyder"] = df["Produktudbyder"].astype(str)
    return df

def annualize_from_monthly(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    ann = (
        df.groupby(["Serie","Produktudbyder","IsBenchmark","Year"])["Return"]
        .apply(lambda s: (1 + s).prod() - 1)
        .reset_index(name="AnnualReturn")
    )
    return ann

def apply_tax(ret: float, tax_rate: float):
    return ret * (1 - tax_rate)

def evolve_balance(annual_returns, start_balance, annual_contrib, contrib_timing="end", tax_rate=0.0):
    balance = start_balance
    balances = []
    for y, r in annual_returns.items():
        r_net = apply_tax(r, tax_rate)
        if contrib_timing == "start":
            balance += annual_contrib
            balance *= (1 + r_net)
        else:
            balance *= (1 + r_net)
            balance += annual_contrib
        balances.append({"Year": int(y), "Return": r, "ReturnNet": r_net, "EndBalance": balance})
    return pd.DataFrame(balances)

def metrics(annual_returns, rf_rate=0.0):
    arr = np.array(list(annual_returns.values()), dtype=float)
    years = len(arr) if len(arr)>0 else np.nan
    if len(arr)==0:
        return {}
    cagr = (1 + arr).prod() ** (1/years) - 1
    vol = np.std(arr, ddof=1) if years>1 else np.nan
    cum = np.cumprod(1 + arr)
    peaks = np.maximum.accumulate(cum)
    dd = (cum - peaks) / peaks
    max_dd = dd.min() if len(dd) else np.nan
    sharpe = (np.mean(arr) - rf_rate) / vol if (vol and vol>0) else np.nan
    return {"CAGR": cagr, "Volatility": vol, "MaxDrawdown": max_dd, "Sharpe": sharpe}

def regression_vs_bench(port_ann_df, bench_ann_df):
    merged = pd.merge(port_ann_df[["Year","AnnualReturn"]], bench_ann_df[["Year","AnnualReturn"]], on="Year", suffixes=("_p","_b"))
    if len(merged) < 3:
        return None
    X = add_constant(merged["AnnualReturn_b"])
    y = merged["AnnualReturn_p"]
    model = OLS(y, X, missing="drop").fit()
    beta = model.params.get("AnnualReturn_b", np.nan)
    alpha = model.params.get("const", np.nan)
    r2 = model.rsquared
    te = np.std(merged["AnnualReturn_p"] - merged["AnnualReturn_b"], ddof=1) if len(merged)>1 else np.nan
    return {"alpha": alpha, "beta": beta, "R2": r2, "tracking_error": te}, merged

DATA_PATH = "afkast_clean.csv"

st.title("Afkast, benchmark & simulering (årlige afkast)")

with st.sidebar:
    st.header("Indstillinger")
    start_balance = st.number_input("Eksisterende opsparing (startværdi)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    horizon_years = st.slider("Horisont (år)", min_value=1, max_value=30, value=10)
    annual_contrib = st.number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")
    contrib_timing = st.selectbox("Indbetalingstidspunkt", options=["start","end"], index=1)
    tax_rate = st.number_input("Effektiv skattesats på afkast", min_value=0.0, max_value=1.0, value=0.15)
    rf_rate = st.number_input("Risikofri rente (årlig)", min_value=-1.0, max_value=1.0, value=0.0)

df = load_data(DATA_PATH)
all_series = sorted(df[~df["IsBenchmark"]]["Serie"].unique().tolist())
serie = st.selectbox("Vælg dataserie (produkt)", options=all_series)

df_s = df[df["Serie"]==serie].copy()
ann = annualize_from_monthly(df_s)

prod_ann = ann[ann["IsBenchmark"]==False].sort_values("Year")
bench_ann = ann[ann["IsBenchmark"]==True].sort_values("Year")

max_year = prod_ann["Year"].max() if not prod_ann.empty else None
if max_year is not None:
    use_years = list(range(int(max_year - horizon_years + 1), int(max_year)+1))
    prod_ann = prod_ann[prod_ann["Year"].isin(use_years)]
    bench_ann = bench_ann[bench_ann["Year"].isin(use_years)]

annual_returns = dict(zip(prod_ann["Year"], prod_ann["AnnualReturn"]))
evo = evolve_balance(annual_returns, start_balance, annual_contrib, contrib_timing, tax_rate)

m = metrics(annual_returns, rf_rate=rf_rate)
reg = regression_vs_bench(prod_ann, bench_ann)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evo["Year"], y=evo["EndBalance"], mode="lines+markers", name="Porteføljeværdi"))
    fig.update_layout(title=f"Værdiudvikling ({serie})", xaxis_title="År", yaxis_title="Værdi")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=prod_ann["Year"], y=prod_ann["AnnualReturn"], name="Produkt"))
    if not bench_ann.empty:
        fig2.add_trace(go.Bar(x=bench_ann["Year"], y=bench_ann["AnnualReturn"], name="Benchmark"))
    fig2.update_layout(title="Årlige afkast", xaxis_title="År", yaxis_title="Afkast")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Nøgletal")
kpi_cols = st.columns(4)
kpi_cols[0].metric("CAGR", f"{m.get('CAGR', np.nan):.2%}" if m else "—")
kpi_cols[1].metric("Volatilitet (årlig)", f"{m.get('Volatility', np.nan):.2%}" if m else "—")
kpi_cols[2].metric("Max drawdown", f"{m.get('MaxDrawdown', np.nan):.2%}" if m else "—")
kpi_cols[3].metric("Sharpe", f"{m.get('Sharpe', np.nan):.2f}" if m else "—")

st.subheader("Benchmarkanalyse")
if reg is not None:
    reg_stats, merged = reg
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta", f"{reg_stats['beta']:.2f}")
    c2.metric("R²", f"{reg_stats['R2']:.2f}")
    c3.metric("Tracking error", f"{reg_stats['tracking_error']:.2%}")
    c4.metric("Alfa", f"{reg_stats['alpha']:.2%}")
    st.dataframe(merged.rename(columns={"AnnualReturn_p":"Produkt","AnnualReturn_b":"Benchmark"}))
else:
    st.info("For få år til regression mod benchmark.")

st.subheader("Resultattabel")
res = prod_ann.merge(bench_ann[["Year","AnnualReturn"]], on="Year", how="left", suffixes=("_Produkt","_Benchmark"))
res = res.merge(evo[["Year","ReturnNet","EndBalance"]], on="Year", how="left")
st.dataframe(res)

st.caption("Note: Skat behandles forenklet som en effektiv sats på årlige afkast. Udvides gerne.")
