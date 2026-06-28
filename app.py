import base64
from pathlib import Path

import requests
import random
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Sovereign Liquidity Lab",
    layout="wide",
)


COUNTRY_URL = "https://api.worldbank.org/v2/country"
APP_DIR = Path(__file__).parent
GLOBE_HEADER_PATH = APP_DIR / "assets" / "global-finance-globe-header.png"
INDICATORS = {
    "FI.RES.TOTL.CD": "reserves_usd",
    "NE.IMP.GNFS.CD": "imports_usd",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "BN.CAB.XOKA.CD": "current_account_usd",
    "NY.GDP.MKTP.CD": "gdp_usd",
    "DT.DOD.DSTC.CD": "short_term_external_debt_usd",
}

FX_TICKERS = {
    "ARG": ("USDARS=X", False),
    "ARM": ("USDAMD=X", False),
    "AUS": ("AUDUSD=X", True),
    "BRA": ("USDBRL=X", False),
    "CAN": ("USDCAD=X", False),
    "CHE": ("USDCHF=X", False),
    "CHL": ("USDCLP=X", False),
    "CHN": ("USDCNY=X", False),
    "COL": ("USDCOP=X", False),
    "CRI": ("USDCRC=X", False),
    "CZE": ("USDCZK=X", False),
    "DOM": ("USDDOP=X", False),
    "EGY": ("USDEGP=X", False),
    "EUR": ("EURUSD=X", True),
    "GBR": ("GBPUSD=X", True),
    "GEO": ("USDGEL=X", False),
    "GHA": ("USDGHS=X", False),
    "HUN": ("USDHUF=X", False),
    "IDN": ("USDIDR=X", False),
    "IND": ("USDINR=X", False),
    "JPN": ("USDJPY=X", False),
    "KAZ": ("USDKZT=X", False),
    "KEN": ("USDKES=X", False),
    "KOR": ("USDKRW=X", False),
    "MAR": ("USDMAD=X", False),
    "MEX": ("USDMXN=X", False),
    "MYS": ("USDMYR=X", False),
    "NGA": ("USDNGN=X", False),
    "NOR": ("USDNOK=X", False),
    "NZL": ("NZDUSD=X", True),
    "PAK": ("USDPKR=X", False),
    "PER": ("USDPEN=X", False),
    "PHL": ("USDPHP=X", False),
    "POL": ("USDPLN=X", False),
    "ROU": ("USDRON=X", False),
    "SWE": ("USDSEK=X", False),
    "THA": ("USDTHB=X", False),
    "TUR": ("USDTRY=X", False),
    "UKR": ("USDUAH=X", False),
    "URY": ("USDUYU=X", False),
    "VNM": ("USDVND=X", False),
    "ZAF": ("USDZAR=X", False),
}

GLOBAL_MARKET_TICKERS = {
    "VIX": "^VIX",
    "US 10Y yield proxy": "^TNX",
    "Oil WTI": "CL=F",
    "Gold": "GC=F",
    "EUR/USD": "EURUSD=X",
}


def image_data_uri(path):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


GLOBE_HEADER_SRC = image_data_uri(GLOBE_HEADER_PATH)


def import_cover_risk_score(x):
    if pd.isna(x):
        return None
    if x >= 5:
        return 0
    if x >= 3:
        return 50
    return 100


def reserves_to_st_debt_risk_score(x):
    if pd.isna(x):
        return None
    if x >= 1.2:
        return 0
    if x >= 1.0:
        return 50
    return 100


def current_account_risk_score(x):
    if pd.isna(x):
        return None
    if x >= -2:
        return 0
    if x >= -5:
        return 50
    return 100


def fx_pressure_risk_score(x):
    if pd.isna(x):
        return None
    if x < 5:
        return 0
    if x < 10:
        return 50
    return 100


def vulnerability_status(x):
    if pd.isna(x):
        return "n/a"
    if x < 30:
        return "low"
    if x < 60:
        return "moderate"
    return "high"


def get_close_series(ticker, period="3y"):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if data.empty:
        return pd.Series(dtype="float64")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.Series(dtype="float64")
        close = data["Close"].iloc[:, 0]
    elif "Close" in data.columns:
        close = data["Close"]
    elif "Adj Close" in data.columns:
        close = data["Adj Close"]
    else:
        return pd.Series(dtype="float64")

    close = close.dropna()
    close.name = ticker
    return close


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_monthly_fx_pressure():
    rows = []

    for country, (ticker, invert) in FX_TICKERS.items():
        close = get_close_series(ticker)
        if close.empty:
            continue

        if invert:
            close = 1 / close

        monthly = close.resample("M").last().dropna()
        if len(monthly) < 13:
            continue

        df = pd.DataFrame(
            {
                "date": monthly.index,
                "country": country,
                "ticker": ticker,
                "fx_rate_local_per_usd": monthly.values,
            }
        )
        df["fx_change_1m"] = df["fx_rate_local_per_usd"].pct_change(1) * 100
        df["fx_change_3m"] = df["fx_rate_local_per_usd"].pct_change(3) * 100
        df["fx_change_12m"] = df["fx_rate_local_per_usd"].pct_change(12) * 100
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    monthly_fx = pd.concat(rows, ignore_index=True)
    monthly_fx["fx_pressure_risk"] = monthly_fx["fx_change_3m"].apply(
        fx_pressure_risk_score
    )
    return monthly_fx


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_global_market_monthly():
    rows = []

    for label, ticker in GLOBAL_MARKET_TICKERS.items():
        close = get_close_series(ticker)
        if close.empty:
            continue
        monthly = close.resample("M").last().dropna()
        if monthly.empty:
            continue
        df = pd.DataFrame(
            {
                "date": monthly.index,
                "indicator": label,
                "ticker": ticker,
                "value": monthly.values,
            }
        )
        df["change_3m"] = df["value"].pct_change(3) * 100
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def get_world_bank_pages(url, params=None):
    params = params or {}
    params = {**params, "format": "json", "per_page": 1000}

    try:
        first = requests.get(url, params=params, timeout=40)
        if first.status_code != 200:
            return []
        payload = first.json()
    except (requests.RequestException, ValueError):
        return []

    if len(payload) < 2 or payload[1] is None:
        return []

    pages = payload[0].get("pages", 1)
    rows = payload[1]

    for page in range(2, pages + 1):
        try:
            page_response = requests.get(
                url,
                params={**params, "page": page},
                timeout=40,
            )
            if page_response.status_code != 200:
                continue
            page_payload = page_response.json()
        except (requests.RequestException, ValueError):
            continue

        if len(page_payload) > 1 and page_payload[1] is not None:
            rows.extend(page_payload[1])

    return rows


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_countries():
    rows = get_world_bank_pages(COUNTRY_URL, params={"per_page": 400})
    countries = pd.DataFrame(rows)
    countries = countries[["id", "iso2Code", "name", "region", "incomeLevel"]].copy()
    countries["region_name"] = countries["region"].apply(lambda x: x["value"])
    countries["income_level"] = countries["incomeLevel"].apply(lambda x: x["value"])
    countries = countries[countries["region_name"] != "Aggregates"].copy()
    countries = countries.rename(columns={"id": "country"})
    return countries[["country", "iso2Code", "name", "region_name", "income_level"]]


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_indicator(indicator_code, indicator_name):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    rows = get_world_bank_pages(url)

    if not rows:
        return pd.DataFrame(columns=["country", "year", indicator_name])

    df = pd.DataFrame(rows)
    if df.empty or "value" not in df.columns:
        return pd.DataFrame(columns=["country", "year", indicator_name])

    df = df[["countryiso3code", "date", "value"]].copy()
    df.columns = ["country", "year", indicator_name]
    df = df.dropna(subset=["country", "year", indicator_name])
    df["year"] = df["year"].astype(int)
    return df


@st.cache_data(ttl=24 * 60 * 60, show_spinner=True)
def build_model():
    countries = load_countries()

    model = None
    for code, name in INDICATORS.items():
        indicator_df = load_indicator(code, name)
        if model is None:
            model = indicator_df
        else:
            model = model.merge(indicator_df, on=["country", "year"], how="outer")

    if model is None or model.empty:
        return pd.DataFrame(), pd.DataFrame()

    model = model.merge(countries, on="country", how="inner")

    model["import_cover_months"] = model["reserves_usd"] / (model["imports_usd"] / 12)
    model["reserves_to_st_debt"] = (
        model["reserves_usd"] / model["short_term_external_debt_usd"]
    )

    model["ca_deficit_need_usd"] = (-model["current_account_usd"]).clip(lower=0)
    model["minimum_reserve_floor_usd"] = model["imports_usd"] / 4
    model["usable_reserves_usd"] = (
        model["reserves_usd"] - model["minimum_reserve_floor_usd"]
    ).clip(lower=0)
    model["baseline_external_financing_need_usd"] = (
        model["ca_deficit_need_usd"] + model["short_term_external_debt_usd"]
    )
    model["baseline_liquidity_gap_usd"] = (
        model["baseline_external_financing_need_usd"] - model["usable_reserves_usd"]
    )
    model["baseline_liquidity_gap_pct_gdp"] = (
        model["baseline_liquidity_gap_usd"] / model["gdp_usd"] * 100
    )

    model["import_cover_risk"] = model["import_cover_months"].apply(
        import_cover_risk_score
    )
    model["reserves_to_st_debt_risk"] = model["reserves_to_st_debt"].apply(
        reserves_to_st_debt_risk_score
    )
    model["current_account_risk"] = model["current_account_gdp"].apply(
        current_account_risk_score
    )

    model["vulnerability_score"] = (
        0.4 * model["import_cover_risk"]
        + 0.4 * model["reserves_to_st_debt_risk"]
        + 0.2 * model["current_account_risk"]
    )
    model["vulnerability_status"] = model["vulnerability_score"].apply(
        vulnerability_status
    )

    latest = (
        model.dropna(subset=["vulnerability_score"])
        .sort_values(["country", "year"])
        .groupby("country")
        .tail(1)
        .copy()
    )

    return model, latest


def format_ranking(df):
    display = df[
        [
            "country",
            "name",
            "region_name",
            "income_level",
            "year",
            "import_cover_months",
            "reserves_to_st_debt",
            "current_account_gdp",
            "baseline_liquidity_gap_pct_gdp",
            "vulnerability_score",
            "vulnerability_status",
        ]
    ].copy()
    display["import_cover_months"] = display["import_cover_months"].round(1)
    display["reserves_to_st_debt"] = display["reserves_to_st_debt"].round(2)
    display["current_account_gdp"] = display["current_account_gdp"].round(1)
    display["baseline_liquidity_gap_pct_gdp"] = display[
        "baseline_liquidity_gap_pct_gdp"
    ].round(1)
    display["vulnerability_score"] = display["vulnerability_score"].round(0)
    return display.sort_values("vulnerability_score", ascending=False)


st.markdown(
    """
    <style>
    :root {
        --navy: #071523;
        --navy-2: #0b1f33;
        --navy-3: #102a43;
        --silver: #d7dde5;
        --titanium: #aeb7c2;
        --platinum: #f2f5f8;
        --accent: #8fa7bf;
        --line: rgba(215, 221, 229, 0.20);
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 0%, rgba(87, 116, 145, 0.28), transparent 32%),
            linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 45%, #050b12 100%);
        color: var(--platinum);
    }

    header[data-testid="stHeader"] {
        background: rgba(7, 21, 35, 0.78);
        border-bottom: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071523 0%, #0b1f33 100%);
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] * {
        color: var(--platinum);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    .sll-hero {
        border: 1px solid rgba(215, 221, 229, 0.22);
        border-radius: 8px;
        padding: 26px 30px;
        margin-bottom: 18px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035)),
            linear-gradient(135deg, rgba(143,167,191,0.16), transparent);
        box-shadow: 0 18px 50px rgba(0,0,0,0.24);
    }

    .sll-kicker {
        color: var(--titanium);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .sll-title {
        color: var(--platinum);
        font-size: 2.8rem;
        font-weight: 760;
        line-height: 1.02;
        margin-bottom: 10px;
    }

    .sll-subtitle {
        color: var(--silver);
        font-size: 1.02rem;
        max-width: 1040px;
        line-height: 1.55;
    }

    .sll-strip {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .sll-chip {
        color: var(--silver);
        border: 1px solid rgba(215, 221, 229, 0.22);
        background: rgba(255,255,255,0.055);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.82rem;
    }

    div[data-testid="stTabs"] button {
        color: var(--silver);
        background: rgba(255,255,255,0.035);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(215, 221, 229, 0.12);
        padding: 12px 16px;
        font-weight: 650;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff;
        background: linear-gradient(180deg, rgba(143,167,191,0.24), rgba(143,167,191,0.08));
        border-bottom: 1px solid rgba(143,167,191,0.65);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(215, 221, 229, 0.16);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--titanium);
    }

    div[data-testid="stMetricValue"] {
        color: var(--platinum);
    }

    h1, h2, h3 {
        color: var(--platinum);
    }

    p, li, label, span {
        color: var(--silver);
    }

    .sll-placeholder {
        border: 1px solid rgba(215, 221, 229, 0.18);
        border-radius: 8px;
        padding: 32px;
        min-height: 340px;
        background: rgba(255,255,255,0.045);
        color: var(--silver);
    }

    .sll-placeholder h3 {
        margin-top: 0;
        color: var(--platinum);
    }

    .stDataFrame {
        border: 1px solid rgba(215, 221, 229, 0.14);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sll-hero">
        <div class="sll-kicker">Sovereign macro-financial surveillance</div>
        <div class="sll-title">Sovereign Liquidity Lab</div>
        <div class="sll-subtitle">
            Public-data analytics for external liquidity, reserve adequacy,
            short-term external debt coverage, and sovereign vulnerability monitoring.
            Prototype only: not a credit rating, investment recommendation, or official assessment.
        </div>
        <div class="sll-strip">
            <div class="sll-chip">Global Data</div>
            <div class="sll-chip">External Liquidity</div>
            <div class="sll-chip">Reserve Adequacy</div>
            <div class="sll-chip">Country Risk Signals</div>
            <div class="sll-chip">Explainable Scoring</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff !important;
        color: #1f2937 !important;
    }

    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e7eb !important;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid #e5e7eb !important;
    }

    section[data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1280px !important;
    }

    .sll-hero {
        display: none !important;
    }

    .sll-topbar {
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-right: calc(50% - 50vw);
        box-sizing: border-box;
        background: linear-gradient(90deg, #0b3d66 0%, #155a8a 54%, #7f9db7 100%);
        border-top: 1px solid #bfd1df;
        border-bottom: 1px solid #9fb6c9;
        border-left: 0;
        border-right: 0;
        border-radius: 0;
        padding: 10px max(24px, calc((100vw - 1280px) / 2 + 24px));
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 18px;
        box-shadow: 0 8px 24px rgba(21, 90, 138, 0.16);
    }

    .sll-brand-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }

    .sll-globe-mark {
        width: 46px;
        height: 46px;
        flex: 0 0 auto;
        border-radius: 50%;
        filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.28));
    }

    .sll-brand {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: 0.02em;
        line-height: 1.15;
    }

    .sll-brand span {
        display: block;
        color: #cbd5e1;
        font-weight: 500;
        margin-top: 3px;
        font-size: 0.86rem;
    }

    .sll-meta {
        color: #edf6ff;
        font-size: 0.82rem;
        white-space: nowrap;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    .sll-footer {
        width: 100vw;
        margin-left: calc(50% - 50vw);
        margin-right: calc(50% - 50vw);
        margin-top: 26px;
        box-sizing: border-box;
        padding: 12px max(24px, calc((100vw - 1280px) / 2 + 24px));
        border-radius: 0;
        background: linear-gradient(90deg, #0b3d66 0%, #155a8a 70%, #7f9db7 100%);
        color: #edf6ff;
        font-size: 0.82rem;
        border-top: 1px solid #bfd1df;
        border-bottom: 1px solid #9fb6c9;
        border-left: 0;
        border-right: 0;
    }

    .sll-footer strong {
        color: #ffffff;
    }
    </style>

    <div class="sll-topbar">
        <div class="sll-brand-wrap">
            <img class="sll-globe-mark" src="__GLOBE_HEADER_SRC__" alt="3D global finance globe">
            <div class="sll-brand">
                Sovereign Liquidity Lab
                <span>External Liquidity Surveillance Terminal</span>
            </div>
        </div>
        <div class="sll-meta">Public data prototype · Global finance data · v0.1</div>
    </div>
    """.replace("__GLOBE_HEADER_SRC__", GLOBE_HEADER_SRC),
    unsafe_allow_html=True,
)

with st.spinner("Loading World Bank data and building global model..."):
    full_model, latest_model = build_model()

if full_model.empty or latest_model.empty:
    st.error(
        "The app could not retrieve enough data from the World Bank API. "
        "Please refresh the app in a minute. If the problem persists, the API may be temporarily unavailable."
    )
    st.stop()

tab_global, tab_profiles, tab_monthly, tab_gap, tab_stress, tab_market, tab_snapshots = st.tabs(
    [
        "Global Dashboard",
        "Country Profiles",
        "Monthly Pressure",
        "Liquidity Gap",
        "Stress Lab",
        "Market Pressure",
        "Country Snapshots",
    ]
)

with tab_global:
    regions = sorted(latest_model["region_name"].dropna().unique())
    statuses = ["low", "moderate", "high"]

    st.markdown("#### Global Dashboard Filters")
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        selected_regions = st.multiselect("Regions", regions, default=regions)
    with filter_col2:
        selected_statuses = st.multiselect(
            "Vulnerability status",
            statuses,
            default=statuses,
        )

    filtered = latest_model[
        latest_model["region_name"].isin(selected_regions)
        & latest_model["vulnerability_status"].isin(selected_statuses)
    ].copy()

    ranking = format_ranking(filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries in View", f"{len(filtered):,.0f}")
    col2.metric("Average Score", f"{filtered['vulnerability_score'].mean():.1f}")
    col3.metric(
        "High Vulnerability",
        f"{(filtered['vulnerability_status'] == 'high').sum():,.0f}",
    )
    col4.metric("Latest Data Year", f"{int(filtered['year'].max())}")

    st.subheader("Global External Vulnerability Ranking")

    top_n = st.slider("Number of countries shown", min_value=10, max_value=60, value=30)
    top = ranking.head(top_n)

    fig = px.bar(
        top.sort_values("vulnerability_score"),
        x="vulnerability_score",
        y="name",
        color="vulnerability_status",
        orientation="h",
        title=f"Top {top_n} Countries by External Vulnerability Score",
        color_discrete_map={
            "low": "#7eb77f",
            "moderate": "#c9a74d",
            "high": "#c85b5b",
            "n/a": "#9e9e9e",
        },
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Vulnerability score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Summary")
    regional_summary = (
        filtered.groupby("region_name")
        .agg(
            countries=("country", "count"),
            avg_score=("vulnerability_score", "mean"),
            median_score=("vulnerability_score", "median"),
            avg_import_cover=("import_cover_months", "mean"),
            avg_reserves_st_debt=("reserves_to_st_debt", "mean"),
        )
        .reset_index()
    )
    regional_summary["avg_score"] = regional_summary["avg_score"].round(1)
    regional_summary["median_score"] = regional_summary["median_score"].round(1)
    regional_summary["avg_import_cover"] = regional_summary["avg_import_cover"].round(1)
    regional_summary["avg_reserves_st_debt"] = regional_summary[
        "avg_reserves_st_debt"
    ].round(2)

    fig_region = px.bar(
        regional_summary.sort_values("avg_score"),
        x="avg_score",
        y="region_name",
        orientation="h",
        title="Average External Vulnerability Score by Region",
        color_discrete_sequence=["#8fa7bf"],
    )
    fig_region.update_layout(
        yaxis_title="",
        xaxis_title="Average vulnerability score",
    )
    st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("Full Ranking Table")
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    csv = ranking.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranking as CSV",
        data=csv,
        file_name="sovereign_liquidity_global_ranking.csv",
        mime="text/csv",
    )

    with st.expander("Methodology and limitations"):
        st.markdown(
            """
            This prototype uses public World Bank data to calculate a transparent external
            vulnerability score.

            Current indicators:

            - **Import cover:** total reserves divided by monthly imports.
            - **Reserves / short-term external debt:** reserve coverage of short-term external debt.
            - **Current account balance:** current account balance as a percent of GDP.
            - **Baseline liquidity gap:** external financing needs minus usable reserves, after preserving
              a three-month import-cover reserve floor.

            Composite score:

            - 40% import-cover risk
            - 40% reserves / short-term external debt risk
            - 20% current-account risk

            The score is a monitoring indicator, not a credit rating. Data availability varies
            across countries and years. Some economies may be excluded when core indicators are missing.
            """
        )


with tab_profiles:
    profile_ranking = format_ranking(latest_model)
    country_names = profile_ranking["name"].sort_values().tolist()

    if (
        "profile_country_name" not in st.session_state
        or st.session_state["profile_country_name"] not in country_names
    ):
        st.session_state["profile_country_name"] = random.choice(country_names)

    header_col, button_col = st.columns([4, 1])
    with header_col:
        st.subheader("Country Profiles")
    with button_col:
        if st.button("Random country", use_container_width=True):
            st.session_state["profile_country_name"] = random.choice(country_names)

    selected_country_name = st.selectbox(
        "Select country",
        country_names,
        index=country_names.index(st.session_state["profile_country_name"]),
        key="country_profile_select",
    )
    st.session_state["profile_country_name"] = selected_country_name

    country_row = profile_ranking[
        profile_ranking["name"] == selected_country_name
    ].iloc[0]

    cols = st.columns(5)
    cols[0].metric("Score", f"{country_row['vulnerability_score']:.0f}")
    cols[1].metric("Status", str(country_row["vulnerability_status"]).upper())
    cols[2].metric("Import Cover", f"{country_row['import_cover_months']:.1f} months")
    cols[3].metric("Reserves/ST Debt", f"{country_row['reserves_to_st_debt']:.2f}x")
    cols[4].metric("CA Balance", f"{country_row['current_account_gdp']:.1f}% GDP")

    country_history = full_model[full_model["country"] == country_row["country"]].copy()
    country_history = country_history.dropna(subset=["vulnerability_score"])

    fig_history = px.line(
        country_history.sort_values("year"),
        x="year",
        y="vulnerability_score",
        title=f"{selected_country_name}: External Vulnerability Score Over Time",
    )
    fig_history.add_hline(y=30, line_dash="dash", line_color="green")
    fig_history.add_hline(y=60, line_dash="dash", line_color="red")
    st.plotly_chart(fig_history, use_container_width=True)

    profile_table = country_history[
        [
            "year",
            "import_cover_months",
            "reserves_to_st_debt",
            "current_account_gdp",
            "baseline_liquidity_gap_pct_gdp",
            "vulnerability_score",
            "vulnerability_status",
        ]
    ].copy()
    profile_table["import_cover_months"] = profile_table["import_cover_months"].round(1)
    profile_table["reserves_to_st_debt"] = profile_table["reserves_to_st_debt"].round(2)
    profile_table["current_account_gdp"] = profile_table["current_account_gdp"].round(1)
    profile_table["baseline_liquidity_gap_pct_gdp"] = profile_table[
        "baseline_liquidity_gap_pct_gdp"
    ].round(1)
    profile_table["vulnerability_score"] = profile_table["vulnerability_score"].round(0)

    st.subheader("Country Indicator History")
    st.dataframe(
        profile_table.sort_values("year", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


with tab_monthly:
    st.subheader("Monthly Pressure")
    st.caption(
        "Monthly market-pressure layer for countries with available free FX tickers. "
        "Positive FX changes mean local-currency depreciation versus the US dollar."
    )

    with st.spinner("Loading monthly FX and global market data..."):
        monthly_fx = load_monthly_fx_pressure()
        global_market = load_global_market_monthly()

    if monthly_fx.empty:
        st.warning(
            "Monthly FX data could not be retrieved. Please refresh the app or try again later."
        )
    else:
        monthly_latest = (
            monthly_fx.dropna(subset=["fx_change_3m"])
            .sort_values(["country", "date"])
            .groupby("country")
            .tail(1)
            .copy()
        )

        monthly_panel = latest_model.merge(
            monthly_latest[
                [
                    "country",
                    "date",
                    "ticker",
                    "fx_rate_local_per_usd",
                    "fx_change_1m",
                    "fx_change_3m",
                    "fx_change_12m",
                    "fx_pressure_risk",
                ]
            ],
            on="country",
            how="inner",
        )

        monthly_panel["monthly_watch_score"] = (
            0.6 * monthly_panel["vulnerability_score"]
            + 0.4 * monthly_panel["fx_pressure_risk"]
        )
        monthly_panel["monthly_watch_status"] = monthly_panel[
            "monthly_watch_score"
        ].apply(vulnerability_status)

        covered = monthly_panel["country"].nunique()
        structural = latest_model["country"].nunique()
        high_fx_pressure = (monthly_panel["fx_pressure_risk"] == 100).sum()
        latest_month = monthly_panel["date"].max().strftime("%b %Y")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly Coverage", f"{covered} / {structural}")
        m2.metric("Latest Month", latest_month)
        m3.metric("High FX Pressure", f"{high_fx_pressure:,.0f}")
        m4.metric(
            "Average 3M FX Move",
            f"{monthly_panel['fx_change_3m'].mean():.1f}%",
        )

        st.markdown("#### Countries With Highest 3M FX Depreciation")
        top_limit = min(50, len(monthly_panel))
        top_min = min(10, top_limit)
        top_default = min(25, top_limit)
        fx_top_n = st.slider(
            "Number of countries shown",
            min_value=top_min,
            max_value=top_limit,
            value=top_default,
            key="monthly_fx_top_n",
        )
        fx_top = monthly_panel.sort_values("fx_change_3m", ascending=False).head(
            fx_top_n
        )

        fig_fx = px.bar(
            fx_top.sort_values("fx_change_3m"),
            x="fx_change_3m",
            y="name",
            color="fx_pressure_risk",
            orientation="h",
            title=f"Top {fx_top_n} Countries by 3M FX Depreciation vs USD",
            color_continuous_scale=["#7eb77f", "#c9a74d", "#c85b5b"],
        )
        fig_fx.add_vline(x=5, line_dash="dash", line_color="orange")
        fig_fx.add_vline(x=10, line_dash="dash", line_color="red")
        fig_fx.update_layout(yaxis_title="", xaxis_title="3M FX change, percent")
        st.plotly_chart(fig_fx, use_container_width=True)

        st.markdown("#### Structural Vulnerability + Monthly FX Pressure")
        watch_top = monthly_panel.sort_values(
            "monthly_watch_score", ascending=False
        ).head(fx_top_n)

        fig_watch = px.bar(
            watch_top.sort_values("monthly_watch_score"),
            x="monthly_watch_score",
            y="name",
            color="monthly_watch_status",
            orientation="h",
            title="Monthly Watch Score",
            color_discrete_map={
                "low": "#7eb77f",
                "moderate": "#c9a74d",
                "high": "#c85b5b",
                "n/a": "#9e9e9e",
            },
        )
        fig_watch.add_vline(x=30, line_dash="dash", line_color="green")
        fig_watch.add_vline(x=60, line_dash="dash", line_color="red")
        fig_watch.update_layout(yaxis_title="", xaxis_title="Monthly watch score")
        st.plotly_chart(fig_watch, use_container_width=True)

        st.markdown("#### Country Monthly FX Trend")
        monthly_country_names = monthly_panel["name"].sort_values().tolist()
        default_monthly_country = (
            "Turkey" if "Turkey" in monthly_country_names else monthly_country_names[0]
        )
        selected_monthly_country = st.selectbox(
            "Select country",
            monthly_country_names,
            index=monthly_country_names.index(default_monthly_country),
            key="monthly_country_select",
        )
        selected_country_code = monthly_panel.loc[
            monthly_panel["name"] == selected_monthly_country, "country"
        ].iloc[0]

        fx_history = monthly_fx[monthly_fx["country"] == selected_country_code].copy()
        fig_country_fx = px.line(
            fx_history.sort_values("date"),
            x="date",
            y="fx_change_3m",
            title=f"{selected_monthly_country}: 3M FX Depreciation vs USD",
        )
        fig_country_fx.add_hline(y=5, line_dash="dash", line_color="orange")
        fig_country_fx.add_hline(y=10, line_dash="dash", line_color="red")
        fig_country_fx.update_layout(xaxis_title="", yaxis_title="3M FX change, percent")
        st.plotly_chart(fig_country_fx, use_container_width=True)

        monthly_table = monthly_panel[
            [
                "country",
                "name",
                "region_name",
                "date",
                "ticker",
                "fx_change_1m",
                "fx_change_3m",
                "fx_change_12m",
                "vulnerability_score",
                "fx_pressure_risk",
                "monthly_watch_score",
                "monthly_watch_status",
            ]
        ].copy()
        monthly_table["date"] = monthly_table["date"].dt.strftime("%Y-%m")
        for col in [
            "fx_change_1m",
            "fx_change_3m",
            "fx_change_12m",
            "vulnerability_score",
            "monthly_watch_score",
        ]:
            monthly_table[col] = monthly_table[col].round(1)

        st.markdown("#### Monthly Coverage Table")
        st.dataframe(
            monthly_table.sort_values("monthly_watch_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        if not global_market.empty:
            st.markdown("#### Global Market Backdrop")
            indicators = sorted(global_market["indicator"].unique())
            selected_global_indicator = st.selectbox(
                "Global indicator",
                indicators,
                key="global_market_indicator_select",
            )
            global_history = global_market[
                global_market["indicator"] == selected_global_indicator
            ].copy()
            fig_global = px.line(
                global_history.sort_values("date"),
                x="date",
                y="value",
                title=f"{selected_global_indicator}: Monthly Level",
            )
            fig_global.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_global, use_container_width=True)


def render_blank_page(title, description):
    st.markdown(
        f"""
        <div class="sll-placeholder">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_gap:
    render_blank_page(
        "Liquidity Gap",
        "Reserved for the external financing needs versus usable reserves module.",
    )

with tab_stress:
    render_blank_page(
        "Stress Lab",
        "Reserved for sudden-stop, reserve-drain, FX-shock, and rollover scenarios.",
    )

with tab_market:
    render_blank_page(
        "Market Pressure",
        "Reserved for FX pressure, global funding conditions, DXY, VIX, rates, and spreads.",
    )

with tab_snapshots:
    render_blank_page(
        "Country Snapshots",
        "Reserved for automated one-page country notes and exportable surveillance briefs.",
    )

st.markdown(
    """
    <div class="sll-footer">
        <strong>Sovereign Liquidity Lab</strong> · Public-data analytical prototype ·
        External liquidity, reserve adequacy, and sovereign vulnerability monitoring ·
        Not a credit rating, investment recommendation, or official assessment.
    </div>
    """,
    unsafe_allow_html=True,
)
