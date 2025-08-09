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
    raw = pd.read_excel(xlsx_path, sheet_name="Afkast")
    date_cols = [c for c in raw.columns if c not in META_COLS]
    long_df = raw.melt(id_vars=META_COLS, value_vars=date_cols, var_name="Date", value_name="Return")

    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df["Return"] = long_df["Return"].astype(str).str.replace(",", ".", regex=False)
    long_df["Return"] = pd.to_numeric(long_df["Return"], errors="coerce")

    if long_df["Return"].dropna().abs().max() > 1.0:
        long_df["Return"] = long_df["Return"] / 100.0

    long_df = long_df.dropna(subset=["Date", "Return"]).reset_index(drop=True)
    long_df["Serie"] = long_df["Name"].astype(str).str.replace(r",?\s*benchmark$", "", regex=True)
    long_df["IsBenchmark"] = long_df["Produkt"].astype(str).str.lower().eq("benchmark")
    return long_df

@st.cache_data
def load_data() -> pd.DataFrame:
    for p in CSV_CANDIDATES:
        if Path(p).exists():
            df = pd.read_csv(p, sep=";", dtype=str)
            df.columns = [c.strip() for c in df.columns]

            if "Date" not in df.columns:
                df = pd.read_csv(p, dtype=str)
                df.columns = [c.strip() for c in df.columns]
            if "Date" not in df.columns:
                raise ValueError("CSV mangler 'Date'.")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Return"] = df["Return"].astype(str).str.replace(",", ".", regex=False)
            df["Return"] = pd.to_numeric(df["Return"], errors="coerce")

            if "Serie" not in df.columns and "Name" in df.columns:
                df["Serie"] = df["Name"].astype(str).str.replace(r",?\s*benchmark$", "", regex=True)
            elif "Serie" not in df.columns:
                raise ValueError("Mangler 'Serie' eller 'Name'.")

            if "IsBenchmark" in df.columns:
                s = df["IsBenchmark"].astype(str).str.strip().str.lower()
                df["IsBenchmark"] = s.isin(["true", "1", "y", "yes"])
            else:
                if "Produkt" in df.columns:
                    df["IsBenchmark"] = df["Produkt"].astype(str).str.lower().eq("benchmark")
                else:
                    df["IsBenchmark"] = False

            if "Produktudbyder" not in df.columns:
                df["Produktudbyder"] = ""

            df = df.dropna(subset=["Date", "Return"]).reset_index(drop=True)
            return df

    for p in XLSX_CANDIDATES:
        if Path(p).exists():
            return _tidy_from_excel(p)

    st.error("Ingen gyldig afkast-fil fundet.")
    st.stop()

def annualize_from_monthly(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Year"] = tmp["Date"].dt.year
    ann = (
        tmp.groupby(["Serie", "Produktudbyder", "IsBenchmark", "Year"])["Return"]
        .apply(lambda s: (1 + s).prod() - 1)
        .reset_index(name="AnnualReturn")
    )
    return ann

def apply_tax(ret: float, tax_rate: float) -> float:
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

# --- UI ---
st.title("Afkast, benchmark & simulering (årlige afkast)")

with st.sidebar:
    st.header("Indstillinger")
    start_balance = st.number_input("Eksisterende opsparing", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    annual_contrib = st.number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")
    contrib_timing = st.selectbox("Indbetalingstidspunkt", options=["start", "end"], index=1)
    tax_rate = st.number_input("Effektiv skattesats", min_value=0.0, max_value=1.0, value=0.15)
    rf_rate = st.number_input("Risikofri rente", min_value=-1.0, max_value=1.0, value=0.0)

df = load_data()

non_bench_mask = ~df["IsBenchmark"].astype(bool)
all_series = sorted(df.loc[non_bench_mask, "Serie"].dropna().unique().tolist())

# 2️⃣ Vælg op til 5 produkter
selected_series = st.multiselect(
    "Vælg op til 5 produkter",
    options=all_series,
    default=all_series[:1],
    max_selections=5
)

# 3️⃣ Periodevalg
years_available = sorted(df["Date"].dt.year.unique())
from_year = st.sidebar.selectbox("Fra år", years_available, index=0)
to_year = st.sidebar.selectbox("Til år", years_available, index=len(years_available)-1)

# Filtrer data
df = df[df["Date"].dt.year.between(from_year, to_year)]
ann = annualize_from_monthly(df)

# Grafer
fig2 = go.Figure()
for serie in selected_series:
    prod_ann = ann[(ann["Serie"] == serie) & (ann["IsBenchmark"] == False)]
    fig2.add_trace(go.Bar(
        x=prod_ann["Year"],
        y=prod_ann["AnnualReturn"] * 100,
        name=serie,
        text=prod_ann["AnnualReturn"].map(lambda x: f"{x:.1%}"),
        textposition='outside'
    ))

fig2.update_layout(title="Årlige afkast (%)", xaxis_title="År", yaxis_title="Afkast (%)")
st.plotly_chart(fig2, use_container_width=True)

# Resultattabel
st.subheader("Resultattabel")
res = ann[ann["Serie"].isin(selected_series) & (ann["IsBenchmark"] == False)].copy()
res["AnnualReturn"] = res["AnnualReturn"].map(lambda x: f"{x:.1%}")
st.dataframe(res)
