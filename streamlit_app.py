# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant

st.set_page_config(page_title="Finans App – Afkast & Simulation", layout="wide")

# --- Konstanter / metadata ---
META_COLS = ["Name", "NHM_Id", "Produktudbyder", "Produkt", "Risikoniveau", "År til pension", "ÅOP"]
CSV_CANDIDATES = ["afkast_clean.csv", "data/afkast_clean.csv"]
XLSX_CANDIDATES = ["Afkast_Delvis.xlsx", "data/Afkast_Delvis.xlsx"]

# --- Indlæsning & klargøring af data ---
def _tidy_from_excel(xlsx_path: str) -> pd.DataFrame:
    # Læs excel og smelt dato-kolonner til lang form
    raw = pd.read_excel(xlsx_path, sheet_name="Afkast")
    date_cols = [c for c in raw.columns if c not in META_COLS]
    long_df = raw.melt(id_vars=META_COLS, value_vars=date_cols, var_name="Date", value_name="Return")

    # Typer + rens
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    # Håndter komma som decimalseparator robust
    long_df["Return"] = (
        long_df["Return"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    long_df["Return"] = pd.to_numeric(long_df["Return"], errors="coerce")

    # Hvis værdier ligner procenter, konverter til decimaler
    if long_df["Return"].dropna().abs().max() > 1.0:
        long_df["Return"] = long_df["Return"] / 100.0

    long_df = long_df.dropna(subset=["Date", "Return"]).reset_index(drop=True)

    # Afledte felter
    long_df["Serie"] = long_df["Name"].astype(str).str.replace(r",?\s*benchmark$", "", regex=True)
    long_df["IsBenchmark"] = long_df["Produkt"].astype(str).str.lower().eq("benchmark")
    return long_df

@st.cache_data
def load_data() -> pd.DataFrame:
    # 1) Prøv CSV (semikolon)
    for p in CSV_CANDIDATES:
        if Path(p).exists():
            df = pd.read_csv(p, sep=";", dtype=str)  # læs som str først for robusthed
            df.columns = [c.strip() for c in df.columns]

            # Tving 'Date' og 'Return' til de rigtige typer
            if "Date" not in df.columns:
                # Hvis CSV er komma-separeret alligevel, prøv igen uden sep
                df = pd.read_csv(p, dtype=str)
                df.columns = [c.strip() for c in df.columns]
            if "Date" not in df.columns:
                raise ValueError("CSV fundet, men indeholder ikke kolonnen 'Date'.")

            # Parse Date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Return som float, håndter evt. komma-decimaler
            if "Return" in df.columns:
                df["Return"] = (
                    df["Return"]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                )
                df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
            else:
                raise ValueError("CSV fundet, men indeholder ikke kolonnen 'Return'.")

            # Sikr 'Serie'
            if "Serie" not in df.columns and "Name" in df.columns:
                df["Serie"] = df["Name"].astype(str).str.replace(r",?\s*benchmark$", "", regex=True)
            elif "Serie" not in df.columns:
                raise ValueError("CSV mangler både 'Serie' og 'Name' (kan ikke aflede serie-navn).")

            # Konverter IsBenchmark til bool robust (tillad True/False/1/0/yes/no)
            if "IsBenchmark" in df.columns:
                s = df["IsBenchmark"].astype(str).str.strip().str.lower()
                df["IsBenchmark"] = s.isin(["true", "1", "y", "yes"])
            else:
                # Fallback: brug 'Produkt' hvis den findes
                if "Produkt" in df.columns:
                    df["IsBenchmark"] = df["Produkt"].astype(str).str.lower().eq("benchmark")
                else:
                    df["IsBenchmark"] = False  # worst case

            # Sikr 'Produktudbyder' findes (kræves nogle steder i UI)
            if "Produktudbyder" not in df.columns:
                df["Produktudbyder"] = ""

            # Rens
            df = df.dropna(subset=["Date", "Return"]).reset_index(drop=True)
            return df

    # 2) Fallback: prøv Excel
    for p in XLSX_CANDIDATES:
        if Path(p).exists():
            return _tidy_from_excel(p)

    # 3) Giv klar besked og stop
    st.error(
        "Fandt hverken en gyldig 'afkast_clean.csv' (semikolon-separeret med 'Date'/'Return') "
        "eller 'Afkast_Delvis.xlsx' i repoets rod."
    )
    st.stop()

def annualize_from_monthly(df: pd.DataFrame) -> pd.DataFrame:
    # Forvent månedlige rater i 'Return' → årlig geometrisk kædning pr. Serie/år
    tmp = df.copy()
    tmp["Year"] = tmp["Date"].dt.year
    ann = (
        tmp.groupby(["Serie", "Produktudbyder", "IsBenchmark", "Year"])["Return"]
        .apply(lambda s: (1 + s).prod() - 1)
        .reset_index(name="AnnualReturn")
    )
    return ann

def apply_tax(ret: float, tax_rate: float) -> float:
    # Forenklet: effektiv skat på årets afkast
    return ret * (1 - tax_rate)

def evolve_balance(annual_returns: dict, start_balance: float, annual_contrib: float,
                   contrib_timing: str = "end", tax_rate: float = 0.0) -> pd.DataFrame:
    balance = start_balance
    rows = []
    for y, r in annual_returns.items():
        r_net = apply_tax(r, tax_rate)
        if contrib_timing == "start":
            balance += annual_contrib
            balance *= (1 + r_net)
        else:
            balance *= (1 + r_net)
            balance += annual_contrib
        rows.append({"Year": int(y), "Return": r, "ReturnNet": r_net, "EndBalance": balance})
    return pd.DataFrame(rows)

def metrics(annual_returns: dict, rf_rate: float = 0.0) -> dict:
    arr = np.array(list(annual_returns.values()), dtype=float)
    if len(arr) == 0:
        return {}
    years = len(arr)
    cagr = (1 + arr).prod() ** (1 / years) - 1
    vol = np.std(arr, ddof=1) if years > 1 else np.nan
    cum = np.cumprod(1 + arr)
    peaks = np.maximum.accumulate(cum)
    dd = (cum - peaks) / peaks
    max_dd = dd.min() if len(dd) else np.nan
    sharpe = (np.mean(arr) - rf_rate) / vol if (vol and vol > 0) else np.nan
    return {"CAGR": cagr, "Volatility": vol, "MaxDrawdown": max_dd, "Sharpe": sharpe}

def regression_vs_bench(port_ann_df: pd.DataFrame, bench_ann_df: pd.DataFrame):
    merged = pd.merge(
        port_ann_df[["Year", "AnnualReturn"]],
        bench_ann_df[["Year", "AnnualReturn"]],
        on="Year",
        suffixes=("_p", "_b")
    )
    if len(merged) < 3:
        return None
    X = add_constant(merged["AnnualReturn_b"])
    y = merged["AnnualReturn_p"]
    model = OLS(y, X, missing="drop").fit()
    beta = model.params.get("AnnualReturn_b", np.nan)
    alpha = model.params.get("const", np.nan)
    r2 = model.rsquared
    te = np.std(merged["AnnualReturn_p"] - merged["AnnualReturn_b"], ddof=1) if len(merged) > 1 else np.nan
    return {"alpha": alpha, "beta": beta, "R2": r2, "tracking_error": te}, merged

# --- UI ---
st.title("Afkast, benchmark & simulering (årlige afkast)")

with st.sidebar:
    st.header("Indstillinger")
    start_balance = st.number_input("Eksisterende opsparing (startværdi)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    horizon_years = st.slider("Horisont (år)", min_value=1, max_value=30, value=10)
    annual_contrib = st.number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")
    contrib_timing = st.selectbox("Indbetalingstidspunkt", options=["start", "end"], index=1)
    tax_rate = st.number_input("Effektiv skattesats på afkast", min_value=0.0, max_value=1.0, value=0.15,
                               help="Forenklet: anvendes direkte på årets afkast.")
    rf_rate = st.number_input("Risikofri rente (årlig)", min_value=-1.0, max_value=1.0, value=0.0)

df = load_data()

# Vælg serie (kun produkter, ikke benchmarks)
if "IsBenchmark" not in df.columns:
    st.error("Data mangler kolonnen 'IsBenchmark'.")
    st.stop()

non_bench_mask = ~df["IsBenchmark"].astype(bool)
all_series = sorted(df.loc[non_bench_mask, "Serie"].dropna().unique().tolist())
if not all_series:
    st.error("Ingen produktserier fundet (kun benchmarks?).")
    st.stop()

serie = st.selectbox("Vælg dataserie (produkt)", options=all_series)

# Filtrér produkt + benchmark for valgt serie
df_s = df[df["Serie"] == serie].copy()
ann = annualize_from_monthly(df_s)

prod_ann = ann[ann["IsBenchmark"] == False].sort_values("Year")
bench_ann = ann[ann["IsBenchmark"] == True].sort_values("Year")

# Afgræns til seneste N år ift. horisont
max_year = prod_ann["Year"].max() if not prod_ann.empty else None
if max_year is not None:
    use_years = list(range(int(max_year - horizon_years + 1), int(max_year) + 1))
    prod_ann = prod_ann[prod_ann["Year"].isin(use_years)]
    bench_ann = bench_ann[bench_ann["Year"].isin(use_years)]

# Udvikling i værdi
annual_returns = dict(zip(prod_ann["Year"], prod_ann["AnnualReturn"]))
evo = evolve_balance(annual_returns, start_balance, annual_contrib, contrib_timing, tax_rate)

# Nøgletal
m = metrics(annual_returns, rf_rate=rf_rate)
reg = regression_vs_bench(prod_ann, bench_ann) if not bench_ann.empty else None

# Grafer
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure()
    if not evo.empty:
        fig.add_trace(go.Scatter(x=evo["Year"], y=evo["EndBalance"], mode="lines+markers", name="Porteføljeværdi"))
    fig.update_layout(title=f"Værdiudvikling ({serie})", xaxis_title="År", yaxis_title="Værdi")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = go.Figure()
    if not prod_ann.empty:
        fig2.add_trace(go.Bar(x=prod_ann["Year"], y=prod_ann["AnnualReturn"], name="Produkt"))
    if not bench_ann.empty:
        fig2.add_trace(go.Bar(x=bench_ann["Year"], y=bench_ann["AnnualReturn"], name="Benchmark"))
    fig2.update_layout(title="Årlige afkast", xaxis_title="År", yaxis_title="Afkast")
    st.plotly_chart(fig2, use_container_width=True)

# KPI-kort
st.subheader("Nøgletal")
kpi_cols = st.columns(4)
kpi_cols[0].metric("CAGR", f"{m.get('CAGR', np.nan):.2%}" if m else "—")
kpi_cols[1].metric("Volatilitet (årlig)", f"{m.get('Volatility', np.nan):.2%}" if m else "—")
kpi_cols[2].metric("Max drawdown", f"{m.get('MaxDrawdown', np.nan):.2%}" if m else "—")
kpi_cols[3].metric("Sharpe", f"{m.get('Sharpe', np.nan):.2f}" if m else "—")

# Benchmarkanalyse
st.subheader("Benchmarkanalyse")
if reg is not None:
    reg_stats, merged = reg
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta", f"{reg_stats['beta']:.2f}")
    c2.metric("R²", f"{reg_stats['R2']:.2f}")
    c3.metric("Tracking error", f"{reg_stats['tracking_error']:.2%}")
    c4.metric("Alfa", f"{reg_stats['alpha']:.2%}")
    st.dataframe(merged.rename(columns={"AnnualReturn_p": "Produkt", "AnnualReturn_b": "Benchmark"}))
else:
    st.info("Ingen eller for få år til regression mod benchmark.")

# Resultattabel
st.subheader("Resultattabel")
res = prod_ann.merge(
    bench_ann[["Year", "AnnualReturn"]],
    on="Year",
    how="left",
    suffixes=("_Produkt", "_Benchmark"),
)
if not evo.empty:
    res = res.merge(evo[["Year", "ReturnNet", "EndBalance"]], on="Year", how="left")
st.dataframe(res)

st.caption("Note: Skat håndteres forenklet som en effektiv sats på årets afkast. Modul kan udvides til præcise regler (pension/aktie/kapital, udbytte vs. kurs, realisation/lager).")
