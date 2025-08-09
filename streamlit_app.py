# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant  # regression

st.set_page_config(page_title="Finans App – Afkast & Simulation", layout="wide")

# ---------------- Persist helpers (avoid pruning when widgets unmount) ----------------
def get_persisted(store_key, default):
    """Return a persisted value; initialize if missing."""
    if store_key not in st.session_state:
        st.session_state[store_key] = default
    return st.session_state[store_key]

def persist_from_widget(widget_key, store_key):
    """If widget updated, copy its value to the persisted key."""
    if widget_key in st.session_state:
        st.session_state[store_key] = st.session_state[widget_key]

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Multiselect chips -> blå */
[data-baseweb="tag"] {
  background-color:#005782 !important; color:#fff !important; border-color:#005782 !important;
}
[data-baseweb="tag"] span{ color:#fff !important; }
[data-baseweb="tag"] svg{ fill:#fff !important; }
[data-baseweb="tag"]:hover,[data-baseweb="tag"]:focus-within{
  background-color:#004667 !important; border-color:#004667 !important;
}
/* Frame the ENTIRE radio (label + options) and place above 'Indstillinger' */
section[data-testid="stSidebar"] div[role="radiogroup"]{
  border:2px solid #005782; border-radius:12px; padding:12px; margin:0 0 14px 0; background:#f6fafc;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  font-weight:700; margin-bottom:8px; color:#0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Constants & help ----------------
META_COLS = ["Name", "NHM_Id", "Produktudbyder", "Produkt", "Risikoniveau", "År til pension", "ÅOP"]
CSV_CANDIDATES = ["afkast_clean.csv", "data/afkast_clean.csv"]
XLSX_CANDIDATES = ["Afkast_Delvis.xlsx", "data/Afkast_Delvis.xlsx"]

KPI_HELP = {
    "CAGR": "Geometrisk gennemsnitligt årligt afkast over perioden.",
    "Volatilitet": "Årlig standardafvigelse af afkast (risikomål).",
    "Max Drawdown": "Største peak-to-trough fald i værdikurven i perioden.",
    "Sharpe Ratio": "Gennemsnitligt merafkast ift. risikofri rente divideret med volatilitet.",
    "Beta": "Hældning i OLS-regression af produktets afkast mod benchmark (følsomhed).",
    "R2": "Forklaret varians fra regressionen (0–1).",
    "TE": "Tracking Error: std. af (produkt − benchmark), årligt.",
    "Alpha": "Intercept i regressionen: merafkast uafhængigt af benchmark (årligt).",
}

# ---------------- Data loading ----------------
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
    # CSV (semicolon or comma)
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

# ---------------- Analysis helpers ----------------
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

# ---------------- Simulation helpers ----------------
def port_params(w_eq, mu_e, mu_b, sig_e, sig_b, rho_eb):
    w_b = 1.0 - w_eq
    mu_p = w_eq * mu_e + w_b * mu_b
    var_p = (w_eq**2)*(sig_e**2) + (w_b**2)*(sig_b**2) + 2*w_eq*w_b*rho_eb*sig_e*sig_b
    sig_p = np.sqrt(max(var_p, 0.0))
    return mu_p, sig_p

def simulate_paths(mu_p, sig_p, years, sims, start, contrib, timing="end", tax_rate=0.0, seed=42):
    rng = np.random.default_rng(seed)
    wealth = np.full(sims, float(start))
    traj = np.empty((years+1, sims), dtype=float)
    traj[0] = wealth
    for t in range(1, years+1):
        if timing == "start":
            wealth += contrib
        r = rng.normal(mu_p, sig_p, sims)
        r_net = r * (1.0 - tax_rate)
        wealth *= (1.0 + r_net)
        if timing == "end":
            wealth += contrib
        traj[t] = wealth
    return traj

def summarize_terminal(traj, p_low=5, p_high=95):
    terminal = traj[-1]
    pl  = float(np.percentile(terminal, p_low))
    p50 = float(np.percentile(terminal, 50))
    ph  = float(np.percentile(terminal, p_high))
    return pl, p50, ph

def summarize_path_percentiles(traj, p_low=5, p_high=95):
    years_axis = np.arange(traj.shape[0])
    pl  = np.percentile(traj, p_low, axis=1)
    p50 = np.percentile(traj, 50,   axis=1)
    ph  = np.percentile(traj, p_high, axis=1)
    return years_axis, pl, p50, ph

def risk_match_required_contrib(target_worstcase, w_eq, mu_e, mu_b, sig_e, sig_b, rho_eb,
                                years, sims, start, timing="end", tax_rate=0.0):
    mu_p, sig_p = port_params(w_eq, mu_e, mu_b, sig_e, sig_b, rho_eb)
    lo, hi = 0.0, max(1.0, start/years * 2 + 100000.0)
    for _ in range(18):  # binær søgning
        mid = 0.5*(lo+hi)
        traj = simulate_paths(mu_p, sig_p, years, sims, start, mid, timing, tax_rate=tax_rate, seed=7)
        pL, _, _ = summarize_terminal(traj)
        if pL < target_worstcase:
            lo = mid
        else:
            hi = mid
    return hi

# ---------------- Sidebar: framed radio ABOVE 'Indstillinger' ----------------
with st.sidebar:
    # Persist mode too
    st.radio("Visning", ["Analyse", "Pensionssimulering"], key="w_mode",
             index=0 if get_persisted("mode", "Analyse") == "Analyse" else 1)
    persist_from_widget("w_mode", "mode")

    st.header("Indstillinger")

mode = st.session_state["mode"]

# ========================================================================
#                                   ANALYSE
# ========================================================================
if mode == "Analyse":
    with st.sidebar:
        # Seed years from data once
        _df_for_years = load_data()
        years_available = sorted(_df_for_years["Date"].dt.year.unique(), reverse=True)
        if years_available:
            st.session_state.setdefault("an_from_year", years_available[0])
            st.session_state.setdefault("an_to_year", years_available[-1])

        # ---- Inputs (widgets use w_*; values mirrored to store keys) ----
        st.number_input("Eksisterende opsparing", min_value=0.0, step=1000.0, format="%.2f",
                        key="w_an_start_balance", value=get_persisted("an_start_balance", 100000.0))
        persist_from_widget("w_an_start_balance", "an_start_balance")

        st.number_input("Årlig indbetaling", min_value=0.0, step=1000.0, format="%.2f",
                        key="w_an_annual_contrib", value=get_persisted("an_annual_contrib", 24000.0))
        persist_from_widget("w_an_annual_contrib", "an_annual_contrib")

        # Fra/Til år indices based on persisted values
        def idx(options, v, fallback=0):
            try: return list(options).index(v)
            except Exception: return fallback

        st.selectbox("Fra år", years_available, key="w_an_from_year",
                     index=idx(years_available, get_persisted("an_from_year", years_available[0]) if years_available else None))
        persist_from_widget("w_an_from_year", "an_from_year")

        st.selectbox("Til år", years_available, key="w_an_to_year",
                     index=idx(years_available, get_persisted("an_to_year", years_available[-1]) if years_available else None,
                               fallback=len(years_available)-1 if years_available else 0))
        persist_from_widget("w_an_to_year", "an_to_year")

        st.selectbox("Indbetalingstidspunkt", ["start", "end"], key="w_an_timing",
                     index=0 if get_persisted("an_timing", "end") == "start" else 1)
        persist_from_widget("w_an_timing", "an_timing")

        st.number_input("Effektiv skattesats", min_value=0.0, max_value=1.0,
                        key="w_an_tax", value=get_persisted("an_tax", 0.15))
        persist_from_widget("w_an_tax", "an_tax")

        st.number_input("Risikofri rente", min_value=-1.0, max_value=1.0,
                        key="w_an_rf", value=get_persisted("an_rf", 0.0))
        persist_from_widget("w_an_rf", "an_rf")

    st.title("Afkast, benchmark & simulering (årlige afkast)")

    df = load_data()
    non_bench_mask = ~df["IsBenchmark"].astype(bool)
    all_series = sorted(df.loc[non_bench_mask, "Serie"].dropna().unique().tolist())

    # Default one product once
    if not st.session_state.get("an_series") and all_series:
        st.session_state["an_series"] = [all_series[0]]

    st.multiselect("Vælg op til 5 produkter", options=all_series, max_selections=5,
                   key="w_an_series", default=st.session_state["an_series"])
    persist_from_widget("w_an_series", "an_series")
    selected_series = st.session_state["an_series"]

    # Use persisted values for calculations
    start_balance  = st.session_state["an_start_balance"]
    annual_contrib = st.session_state["an_annual_contrib"]
    from_year      = st.session_state["an_from_year"]
    to_year        = st.session_state["an_to_year"]
    contrib_timing = st.session_state["an_timing"]
    tax_rate       = st.session_state["an_tax"]
    rf_rate        = st.session_state["an_rf"]

    # Filter and aggregate
    year_min, year_max = sorted([from_year, to_year])
    df = df[df["Date"].dt.year.between(year_min, year_max)]
    ann = annualize_from_monthly(df)

    # Årlige afkast
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

    # Værdiudvikling + KPI’er
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

        bench_ann = ann[(ann["Serie"] == serie) & (ann["IsBenchmark"])]
        reg_stats, _ = regression_vs_bench(prod_ann, bench_ann)
        if reg_stats is not None:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Beta", f"{reg_stats['beta']:.2f}", help=KPI_HELP["Beta"])
            d2.metric("R²", f"{reg_stats['R2']:.2f}", help=KPI_HELP["R2"])
            d3.metric("Tracking Error", f"{reg_stats['TE']*100:.1f}%", help=KPI_HELP["TE"])
            d4.metric("Alfa", f"{reg_stats['alpha']*100:.1f}%", help=KPI_HELP["Alpha"])
        else:
            st.info(f"{serie}: For få år til benchmark-regression.")

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

    st.subheader("Resultattabel")
    res = ann[ann["Serie"].isin(selected_series) & (~ann["IsBenchmark"])].copy()
    res["AnnualReturn"] = res["AnnualReturn"].map(lambda x: f"{x:.1%}")
    st.dataframe(res)

# ========================================================================
#                             PENSIONS-SIMULERING
# ========================================================================
else:
    st.title("Pensionssimulering – Aktier vs. Obligationer")

    c1, c2, c3 = st.columns(3)
    c1.number_input("Startkapital", min_value=0.0, step=10000.0, format="%.0f",
                    key="w_sim_start_cap", value=get_persisted("sim_start_cap", 250000.0))
    persist_from_widget("w_sim_start_cap", "sim_start_cap")

    c2.number_input("Årlig indbetaling", min_value=0.0, step=1000.0, format="%.0f",
                    key="w_sim_yearly_contrib", value=get_persisted("sim_yearly_contrib", 24000.0))
    persist_from_widget("w_sim_yearly_contrib", "sim_yearly_contrib")

    c3.number_input("Investeringshorisont (år)", min_value=1, step=1,
                    key="w_sim_horizon", value=get_persisted("sim_horizon", 18))
    persist_from_widget("w_sim_horizon", "sim_horizon")

    st.selectbox("Indbetalingstidspunkt", ["start", "end"], key="w_sim_timing",
                 index=0 if get_persisted("sim_timing", "end") == "start" else 1)
    persist_from_widget("w_sim_timing", "sim_timing")

    st.slider("Antal simulationer", 1000, 20000, step=1000, key="w_sim_nsims",
              value=get_persisted("sim_nsims", 5000),
              help="Flere simulationer = mere stabile percentiler, men langsommere.")
    persist_from_widget("w_sim_nsims", "sim_nsims")

    # Global skat (in sidebar) + toggle
    tax_rate_global = st.sidebar.number_input("Effektiv skattesats (global)", min_value=0.0, max_value=1.0,
                                              key="w_sim_tax_global", value=get_persisted("sim_tax_global", 0.15),
                                              help="Effektiv sats pr. år – samme logik som i Analysen.")
    persist_from_widget("w_sim_tax_global", "sim_tax_global")

    st.checkbox("Anvend skat i simuleringen (samme sats som global)",
                key="w_sim_use_tax", value=get_persisted("sim_use_tax", True))
    persist_from_widget("w_sim_use_tax", "sim_use_tax")
    tax_rate_sim = st.session_state["sim_tax_global"] if st.session_state["sim_use_tax"] else 0.0

    st.selectbox("Percentiler", ["5/95", "10/90"], key="w_sim_pct_preset",
                 index=0 if get_persisted("sim_pct_preset", "5/95") == "5/95" else 1)
    persist_from_widget("w_sim_pct_preset", "sim_pct_preset")
    p_low, p_high = (5, 95) if st.session_state["sim_pct_preset"] == "5/95" else (10, 90)

    st.markdown("### Porteføljevægte")
    wc1, wc2 = st.columns(2)
    wc1.slider("Primær portefølje – Aktieandel (%)", 0, 100, step=5, key="w_sim_w_eq_primary",
               value=get_persisted("sim_w_eq_primary", 60))
    persist_from_widget("w_sim_w_eq_primary", "sim_w_eq_primary")

    wc2.slider("Alternativ portefølje – Aktieandel (%)", 0, 100, step=5, key="w_sim_w_eq_alt",
               value=get_persisted("sim_w_eq_alt", 80))
    persist_from_widget("w_sim_w_eq_alt", "sim_w_eq_alt")

    with st.expander("Avancerede antagelser (afkast/vol/korrelation)"):
        a1, a2, a3 = st.columns(3)
        a1.number_input("Forventet årligt aktieafkast", step=0.005, format="%.3f",
                        key="w_sim_mu_eq", value=get_persisted("sim_mu_eq", 0.07))
        a2.number_input("Forventet årligt obligationsafkast", step=0.005, format="%.3f",
                        key="w_sim_mu_bd", value=get_persisted("sim_mu_bd", 0.02))
        a3.number_input("Korrelation (aktier, obligationer)", min_value=-1.0, max_value=1.0, step=0.05,
                        key="w_sim_rho", value=get_persisted("sim_rho", 0.10))
        persist_from_widget("w_sim_mu_eq", "sim_mu_eq")
        persist_from_widget("w_sim_mu_bd", "sim_mu_bd")
        persist_from_widget("w_sim_rho",   "sim_rho")

        b1, b2 = st.columns(2)
        b1.number_input("Aktie-volatilitet (årlig std.)", step=0.01, format="%.2f",
                        key="w_sim_sig_eq", value=get_persisted("sim_sig_eq", 0.18))
        b2.number_input("Obligations-volatilitet (årlig std.)", step=0.01, format="%.2f",
                        key="w_sim_sig_bd", value=get_persisted("sim_sig_bd", 0.06))
        persist_from_widget("w_sim_sig_eq", "sim_sig_eq")
        persist_from_widget("w_sim_sig_bd", "sim_sig_bd")

    # Pull persisted values for simulation
    start_cap      = st.session_state["sim_start_cap"]
    yearly_contrib = st.session_state["sim_yearly_contrib"]
    horizon_years  = st.session_state["sim_horizon"]
    contrib_timing = st.session_state["sim_timing"]
    n_sims         = st.session_state["sim_nsims"]
    w_eq_primary   = st.session_state["sim_w_eq_primary"] / 100.0
    w_eq_alt       = st.session_state["sim_w_eq_alt"] / 100.0
    mu_eq          = st.session_state["sim_mu_eq"]
    mu_bd          = st.session_state["sim_mu_bd"]
    rho            = st.session_state["sim_rho"]
    sig_eq         = st.session_state["sim_sig_eq"]
    sig_bd         = st.session_state["sim_sig_bd"]

    # Porteføljeparametre
    mu_p_primary, sig_p_primary = port_params(w_eq_primary, mu_eq, mu_bd, sig_eq, sig_bd, rho)
    mu_p_alt,     sig_p_alt     = port_params(w_eq_alt,     mu_eq, mu_bd, sig_eq, sig_bd, rho)

    # Simulationer
    traj_primary = simulate_paths(mu_p_primary, sig_p_primary, horizon_years, n_sims,
                                  start_cap, yearly_contrib, contrib_timing, tax_rate=tax_rate_sim, seed=1)
    traj_alt     = simulate_paths(mu_p_alt,     sig_p_alt,     horizon_years, n_sims,
                                  start_cap, yearly_contrib, contrib_timing, tax_rate=tax_rate_sim, seed=2)

    # Resultater (terminal)
    pL_p, p50_p, pH_p = summarize_terminal(traj_primary, p_low, p_high)
    pL_a, p50_a, pH_a = summarize_terminal(traj_alt,     p_low, p_high)

    st.subheader("Resultater: Primær portefølje")
    c1, c2, c3 = st.columns(3)
    c1.metric("Forventet depot (median)", f"{p50_p:,.0f} kr")
    c2.metric(f"{p_low}% worst-case", f"{pL_p:,.0f} kr")
    c3.metric(f"{p_high}% best-case", f"{pH_p:,.0f} kr")

    st.subheader("Resultater: Alternativ portefølje")
    d1, d2, d3 = st.columns(3)
    d1.metric("Forventet depot (median)", f"{p50_a:,.0f} kr")
    d2.metric(f"{p_low}% worst-case", f"{pL_a:,.0f} kr")
    d3.metric(f"{p_high}% best-case", f"{pH_a:,.0f} kr")

    st.subheader("Ændringer ved alternativ")
    e1, e2, e3 = st.columns(3)
    e1.metric("Forventet gevinst", f"{(p50_a - p50_p):,.0f} kr")
    e2.metric("Risiko (worst-case) ændring", f"{(pL_a - pL_p):,.0f} kr")
    e3.metric("Upside (best-case) ændring", f"{(pH_a - pH_p):,.0f} kr")

    # Risk matching (match samme low-percentil som primær)
    st.markdown("### 🔁 Matching af risiko")
    req_contrib = risk_match_required_contrib(
        target_worstcase=pL_p,
        w_eq=w_eq_alt,
        mu_e=mu_eq, mu_b=mu_bd, sig_e=sig_eq, sig_b=sig_bd, rho_eb=rho,
        years=horizon_years, sims=n_sims, start=start_cap, timing=contrib_timing, tax_rate=tax_rate_sim
    )
    traj_matched = simulate_paths(mu_p_alt, sig_p_alt, horizon_years, n_sims,
                                  start_cap, req_contrib, contrib_timing, tax_rate=tax_rate_sim, seed=3)
    _, p50_m, _ = summarize_terminal(traj_matched, p_low, p_high)

    st.write(f"For at fastholde samme worst-case ({pL_p:,.0f} kr), bør årlig indbetaling hæves til **{req_contrib:,.0f} kr**.")
    st.write(f"Det giver en forventet opsparing på **{p50_m:,.0f} kr**, altså en gevinst på **{(p50_m - p50_p):,.0f} kr** i forhold til primær.")

    # Udvikling over tid (percentiler)
    st.markdown("### Udvikling over tid (percentiler)")
    years_axis, pLp, p50p, pHp = summarize_path_percentiles(traj_primary, p_low, p_high)
    _,          pLa, p50a, pHa = summarize_path_percentiles(traj_alt,     p_low, p_high)

    fig = go.Figure()
    # Primær
    fig.add_trace(go.Scatter(x=years_axis, y=p50p, mode="lines", name="Primær – median", line=dict(color="#005782", width=3)))
    fig.add_trace(go.Scatter(x=years_axis, y=pLp,  mode="lines", name=f"Primær – {p_low}%",  line=dict(color="#005782", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=years_axis, y=pHp,  mode="lines", name=f"Primær – {p_high}%", line=dict(color="#005782", width=1, dash="dot")))
    # Alternativ
    fig.add_trace(go.Scatter(x=years_axis, y=p50a, mode="lines", name="Alternativ – median", line=dict(color="#ff6600", width=3)))
    fig.add_trace(go.Scatter(x=years_axis, y=pLa,  mode="lines", name=f"Alternativ – {p_low}%",  line=dict(color="#ff6600", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=years_axis, y=pHa,  mode="lines", name=f"Alternativ – {p_high}%", line=dict(color="#ff6600", width=1, dash="dot")))

    fig.update_layout(
        xaxis_title="År",
        yaxis_title="Depot (kr.)",
        hovermode="x unified",
        legend_title_text="Portefølje / percentil"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Simulationen bruger normalfordelte årlige afkast. Hvis 'Anvend skat' er slået til, reduceres hvert års afkast med den valgte effektive sats.")
