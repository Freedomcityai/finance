# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant  # <-- til regression

st.set_page_config(page_title="Finans App – Afkast & Simulation", layout="wide")

META_COLS = ["Name", "NHM_Id", "Produktudbyder", "Produkt", "Risikoniveau", "År til pension", "ÅOP"]
CSV_CANDIDATES = ["afkast_clean.csv", "data/afkast_clean.csv"]
XLSX_CANDIDATES = ["Afkast_Delvis.xlsx", "data/Afkast_Delvis.xlsx"]

# --- KPI tooltips ---
KPI_HELP = {
    "CAGR": "Geometrisk gennemsnitligt årligt afkast over perioden.",
    "Volatilitet": "Årlig standardafvigelse af afkast (risikomål).",
    "Max Drawdown": "Største peak-to-trough fald i værdikurven i perioden.",
    "Sharpe Ratio": "Gennemsnitligt merafkast ift. risikofri rente divideret med volatilitet.",
    # Benchmark-KPI'er:
    "Beta": "Hældningen i en OLS-regression af produktets afkast mod benchmark (følsomhed).",
    "R2": "Forklaret varians fra regressionen (0–1).",
    "TE": "Tracking Error: standardafvigelsen af (produkt − benchmark), årlig.",
    "Alpha": "Interceptet i regressionen: merafkast uafhængigt af benchmark (årligt).",
}

# --- DATA LOADING ---
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
    # CSV (semikolon eller komma)
    for p in CSV_CANDIDATES:
        if Path(p).exists():
            df = pd.read_csv(p, sep=";", dtype=str)
            df.columns = [c.strip() for c in df.columns]
            if "Date" not in df.columns:  # prøv komma-separeret
                df = pd.read_csv(p, dtype=str)
                df.columns = [c.strip() for c in df.columns]
            if "Date" not in df.columns:
                raise ValueError("CSV mangler 'Date'.")

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Return"] = (df["Return"].astype(str).str.replace(",", ".", regex=False))
            df["Return"] = pd.to_numeric(df["Return"], errors="coerce")

            if "Serie" not in df.columns and "Name" in df.columns:
                df["Serie"] = df["Name"].astype(str).str.replace(r",?\s*benchmark$", "", regex=True)
            if "IsBenchmark" in df.columns:
                s = df["IsBenchmark"].astype(str).str.strip().str.lower()
                df["IsBenchmark"] = s.isin(["true", "1", "y", "yes"])
            elif "Produkt" in df.columns:
                df["IsBenchmark"] = df["Produkt"].astype(str).str.lower().eq("benchmark")
            else:
                df["IsBenchmark"] = False

            if "Produktudbyder" not in df.columns:
                df["Produktudbyder"] = ""

            df = df.dropna(subset=["Date", "Return"]).reset_index(drop=True)
            return df

    # Excel fallback
    for p in XLSX_CANDIDATES:
        if Path(p).exists():
            return _tidy_from_excel(p)

    st.error("Ingen gyldig afkast-fil fundet (afkast_clean.csv eller Afkast_Delvis.xlsx).")
    st.stop()

# --- CALCULATIONS ---
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

def regression_vs_bench(prod_ann: pd.DataFrame, bench_ann: pd.DataFrame):
    """OLS: produkt = alpha + beta*benchmark. Returnér alpha, beta, R2, tracking error + merged data."""
    merged = pd.merge(
        prod_ann[["Year", "AnnualReturn"]],
        bench_ann[["Year", "AnnualReturn"]],
        on="Year",
        suffixes=("_p", "_b"),
    )
    if len(merged) < 3:
        return None, None
    X = add_constant(merged["AnnualReturn_b"])
    y = merged["AnnualReturn_p"]
    model = OLS(y, X, missing="drop").fit()
    beta = model.params.get("AnnualReturn_b", np.nan)
    alpha = model.params.get("const", np.nan)
    r2 = model.rsquared
    te = np.std(merged["AnnualReturn_p"] - merged["AnnualReturn_b"], ddof=1) if len(merged) > 1 else np.nan
    return {"beta": beta, "alpha": alpha, "R2": r2, "TE": te}, merged

# --- UI ---
st.title("Afkast, benchmark & simulering (årlige afkast)")

with st.sidebar:
    st.header("Indstillinger")
    start_balance = st.number_input("Eksisterende opsparing", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    annual_contrib = st.number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")

    # Årsfiltre – faldende sortering
    _df_for_years = load_data()
    years_available = sorted(_df_for_years["Date"].dt.year.unique(), reverse=True)
    from_year = st.selectbox("Fra år", years_available, index=0)
    to_year   = st.selectbox("Til år",  years_available, index=len(years_available)-1)

    contrib_timing = st.selectbox("Indbetalingstidspunkt", options=["start", "end"], index=1)
    tax_rate = st.number_input("Effektiv skattesats", min_value=0.0, max_value=1.0, value=0.15)
    rf_rate = st.number_input("Risikofri rente", min_value=-1.0, max_value=1.0, value=0.0)

# Indlæs rigtige data
df = load_data()

# Produktvalg (max 5, kun non-benchmark)
non_bench_mask = ~df["IsBenchmark"].astype(bool)
all_series = sorted(df.loc[non_bench_mask, "Serie"].dropna().unique().tolist())
selected_series = st.multiselect(
    "Vælg op til 5 produkter",
    options=all_series,
    default=all_series[:1],
    max_selections=5
)

# Filtrer på periode
year_min, year_max = sorted([from_year, to_year])
df = df[df["Date"].dt.year.between(year_min, year_max)]
ann = annualize_from_monthly(df)

# Årlige afkast – samlet søjlediagram (1 decimal, i %)
fig2 = go.Figure()
for serie in selected_series:
    prod_ann = ann[(ann["Serie"] == serie) & (~ann["IsBenchmark"])]
    fig2.add_trace(go.Bar(
        x=prod_ann["Year"],
        y=prod_ann["AnnualReturn"] * 100,
        name=serie,
        text=prod_ann["AnnualReturn"].map(lambda x: f"{x:.1%}"),
        textposition='outside'
    ))
fig2.update_layout(title="Årlige afkast (%)", xaxis_title="År", yaxis_title="Afkast (%)")
st.plotly_chart(fig2, use_container_width=True)

# Værdiudvikling – fælles graf + individuelle KPI'er (inkl. benchmark-KPI'er)
st.subheader("Værdiudvikling & KPI'er")
fig_bal = go.Figure()

for serie in selected_series:
    prod_ann = ann[(ann["Serie"] == serie) & (~ann["IsBenchmark"])]
    ann_dict = dict(zip(prod_ann["Year"], prod_ann["AnnualReturn"]))

    bal_df = evolve_balance(
        annual_returns=ann_dict,
        start_balance=start_balance,
        annual_contrib=annual_contrib,
        contrib_timing=contrib_timing,
        tax_rate=tax_rate
    )

    met = metrics(ann_dict, rf_rate)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{serie} CAGR", f"{met['CAGR']*100:.1f}%", help=KPI_HELP["CAGR"])
    c2.metric("Volatilitet", f"{met['Volatility']*100:.1f}%", help=KPI_HELP["Volatilitet"])
    c3.metric("Max Drawdown", f"{met['MaxDrawdown']*100:.1f}%", help=KPI_HELP["Max Drawdown"])
    c4.metric("Sharpe Ratio", f"{met['Sharpe']:.2f}", help=KPI_HELP["Sharpe Ratio"])

    # Benchmark-regression for serien
    bench_ann = ann[(ann["Serie"] == serie) & (ann["IsBenchmark"])]
    reg_stats, _merged = regression_vs_bench(prod_ann, bench_ann)
    if reg_stats is not None:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Beta", f"{reg_stats['beta']:.2f}", help=KPI_HELP["Beta"])
        d2.metric("R²", f"{reg_stats['R2']:.2f}", help=KPI_HELP["R2"])
        d3.metric("Tracking Error", f"{reg_stats['TE']*100:.1f}%", help=KPI_HELP["TE"])
        d4.metric("Alfa", f"{reg_stats['alpha']*100:.1f}%", help=KPI_HELP["Alpha"])
    else:
        st.info(f"{serie}: For få år til benchmark-regression.")

    # Til fælles værdiudviklingsgraf
    fig_bal.add_trace(go.Scatter(
        x=bal_df["Year"],
        y=bal_df["EndBalance"],
        mode="lines+markers",
        name=serie
    ))

fig_bal.update_layout(
    title="Værdiudvikling – alle valgte produkter",
    xaxis_title="År",
    yaxis_title="Balance (kr.)"
)
st.plotly_chart(fig_bal, use_container_width=True)

# Resultattabel (formatteret i %)
st.subheader("Resultattabel")
res = ann[ann["Serie"].isin(selected_series) & (~ann["IsBenchmark"])].copy()
res["AnnualReturn"] = res["AnnualReturn"].map(lambda x: f"{x:.1%}")
st.dataframe(res)
