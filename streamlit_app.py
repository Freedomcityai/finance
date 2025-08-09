# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant  # regression

st.set_page_config(page_title="Finans App – Afkast & Simulation", layout="wide")
st.markdown("""
<style>
/* Framed radio group at top of sidebar */
section[data-testid="stSidebar"] .mode-box{
  border:2px solid #005782; border-radius:12px; padding:14px; margin-bottom:14px;
}
section[data-testid="stSidebar"] .mode-title{
  font-weight:600; margin-bottom:8px; color:#0f172a;
}
</style>
""", unsafe_allow_html=True)


# --- Farv valgte tags i multiselect (#005782) ---
st.markdown("""
<style>
[data-baseweb="tag"] {
    background-color: #005782 !important;
    color: white !important;
    border-color: #005782 !important;
}
[data-baseweb="tag"] span { color: white !important; }
[data-baseweb="tag"] svg  { fill: white !important; }
[data-baseweb="tag"]:hover,
[data-baseweb="tag"]:focus-within {
    background-color: #004667 !important;
    border-color: #004667 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Konstanter & hjælpetekster ----------
META_COLS = ["Name", "NHM_Id", "Produktudbyder", "Produkt", "Risikoniveau", "År til pension", "ÅOP"]
CSV_CANDIDATES = ["afkast_clean.csv", "data/afkast_clean.csv"]
XLSX_CANDIDATES = ["Afkast_Delvis.xlsx", "data/Afkast_Delvis.xlsx"]

KPI_HELP = {
    "CAGR": "Geometrisk gennemsnitligt årligt afkast over perioden.",
    "Volatilitet": "Årlig standardafvigelse af afkast (risikomål).",
    "Max Drawdown": "Største peak-to-trough fald i værdikurven i perioden.",
    "Sharpe Ratio": "Gennemsnitligt merafkast ift. risikofri rente divideret med volatilitet.",
    "Beta": "Hældningen i en OLS-regression af produktets afkast mod benchmark (følsomhed).",
    "R2": "Forklaret varians fra regressionen (0–1).",
    "TE": "Tracking Error: standardafvigelsen af (produkt − benchmark), årlig.",
    "Alpha": "Intercept i regressionen: merafkast uafhængigt af benchmark (årligt).",
}

# ---------- DATA LOADING ----------
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
            if "Date" not in df.columns:  # prøv komma
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

# ---------- BEREGNINGER (Analyse) ----------
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
    peaks = np.maximum.accumulate(cum)  # fix
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

# ---------- BEREGNINGER (Simulation) ----------
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
        r_net = r * (1.0 - tax_rate)     # effektiv skat pr. år
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
    """Find årlig indbetaling der giver samme low-percentil som target_worstcase."""
    mu_p, sig_p = port_params(w_eq, mu_e, mu_b, sig_e, sig_b, rho_eb)
    lo, hi = 0.0, max(1.0, start/years * 2 + 100000.0)
    for _ in range(18):  # binær søgning
        mid = 0.5*(lo+hi)
        traj = simulate_paths(mu_p, sig_p, years, sims, start, mid, timing, tax_rate=tax_rate, seed=7)
        pL, _, _ = summarize_terminal(traj)  # samme p_low antages som i kaldet til funktionen
        if pL < target_worstcase:
            lo = mid
        else:
            hi = mid
    return hi

# ---------- Sidebar: vælg visning ----------
st.sidebar.header("Indstillinger")
with st.sidebar:
    st.markdown('<div class="mode-box"><div class="mode-title">Visning</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Analyse", "Pensionssimulering"], key="mode", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)  # close .mode-box

# =========================================================
# ======================== ANALYSE ========================
# =========================================================
if mode == "Analyse":
    start_balance = st.sidebar.number_input("Eksisterende opsparing", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    annual_contrib = st.sidebar.number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")

    # Årsfiltre – faldende sortering (placeret her, lige efter indbetaling)
    _df_for_years = load_data()
    years_available = sorted(_df_for_years["Date"].dt.year.unique(), reverse=True)
    from_year = st.sidebar.selectbox("Fra år", years_available, index=0)
    to_year   = st.sidebar.selectbox("Til år",  years_available, index=len(years_available)-1)

    contrib_timing = st.sidebar.selectbox("Indbetalingstidspunkt", options=["start", "end"], index=1)
    tax_rate = st.sidebar.number_input("Effektiv skattesats", min_value=0.0, max_value=1.0, value=0.15)
    rf_rate  = st.sidebar.number_input("Risikofri rente", min_value=-1.0, max_value=1.0, value=0.0)

    st.title("Afkast, benchmark & simulering (årlige afkast)")

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

        # Benchmark-KPI’er
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

# =========================================================
# =================== PENSIONS-SIMULERING =================
# =========================================================
else:
    st.title("Pensionssimulering – Aktier vs. Obligationer")

    # Inputs (basis)
    sim_cols = st.columns(3)
    start_cap      = sim_cols[0].number_input("Startkapital", min_value=0.0, value=250000.0, step=10000.0, format="%.0f")
    yearly_contrib = sim_cols[1].number_input("Årlig indbetaling", min_value=0.0, value=24000.0, step=1000.0, format="%.0f")
    horizon_years  = sim_cols[2].number_input("Investeringshorisont (år)", min_value=1, value=18, step=1)

    contrib_timing = st.selectbox("Indbetalingstidspunkt", ["start", "end"], index=1)
    n_sims = st.slider("Antal simulationer", 1000, 20000, 5000, step=1000, help="Flere simulationer = mere stabile percentiler, men langsommere.")

    # Skat i simuleringen
    tax_rate_global = st.sidebar.number_input("Effektiv skattesats (global)", min_value=0.0, max_value=1.0, value=0.15, help="Samme logik som i Analysen.")
    use_tax_sim = st.checkbox("Anvend skat i simuleringen (samme sats som global)", value=True)
    tax_rate_sim = tax_rate_global if use_tax_sim else 0.0

    # Percentilvalg
    preset = st.selectbox("Percentiler", ["5/95", "10/90"], index=0)
    if preset == "5/95":
        p_low, p_high = 5, 95
    else:
        p_low, p_high = 10, 90

    # Vægte
    st.markdown("### Porteføljevægte")
    weights_cols = st.columns(2)
    w_eq_primary = weights_cols[0].slider("Primær portefølje – Aktieandel (%)", 0, 100, 60, step=5) / 100.0
    w_eq_alt     = weights_cols[1].slider("Alternativ portefølje – Aktieandel (%)", 0, 100, 80, step=5) / 100.0

    with st.expander("Avancerede antagelser (afkast/vol/korrelation)"):
        a1, a2, a3 = st.columns(3)
        mu_eq = a1.number_input("Forventet årligt aktieafkast", value=0.07, step=0.005, format="%.3f")
        mu_bd = a2.number_input("Forventet årligt obligationsafkast", value=0.02, step=0.005, format="%.3f")
        rho   = a3.number_input("Korrelation (aktier, obligationer)", value=0.10, step=0.05, min_value=-1.0, max_value=1.0)

        b1, b2 = st.columns(2)
        sig_eq = b1.number_input("Aktie-volatilitet (årlig std.)", value=0.18, step=0.01, format="%.2f")
        sig_bd = b2.number_input("Obligations-volatilitet (årlig std.)", value=0.06, step=0.01, format="%.2f")

    # Beregn porteføljeparametre
    mu_p_primary, sig_p_primary = port_params(w_eq_primary, mu_eq, mu_bd, sig_eq, sig_bd, rho)
    mu_p_alt,     sig_p_alt     = port_params(w_eq_alt,     mu_eq, mu_bd, sig_eq, sig_bd, rho)

    # Simulationer
    traj_primary = simulate_paths(mu_p_primary, sig_p_primary, horizon_years, n_sims,
                                  start_cap, yearly_contrib, contrib_timing,
                                  tax_rate=tax_rate_sim, seed=1)
    traj_alt     = simulate_paths(mu_p_alt,     sig_p_alt,     horizon_years, n_sims,
                                  start_cap, yearly_contrib, contrib_timing,
                                  tax_rate=tax_rate_sim, seed=2)

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

    # Risk matching (match samme worst-case som primær ved p_low)
    st.markdown("### 🔁 Matching af risiko")
    req_contrib = risk_match_required_contrib(
        target_worstcase=pL_p,
        w_eq=w_eq_alt,
        mu_e=mu_eq, mu_b=mu_bd, sig_e=sig_eq, sig_b=sig_bd, rho_eb=rho,
        years=horizon_years, sims=n_sims, start=start_cap, timing=contrib_timing,
        tax_rate=tax_rate_sim
    )
    traj_matched = simulate_paths(mu_p_alt, sig_p_alt, horizon_years, n_sims,
                                  start_cap, req_contrib, contrib_timing,
                                  tax_rate=tax_rate_sim, seed=3)
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
