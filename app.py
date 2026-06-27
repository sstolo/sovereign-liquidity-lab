import requests
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Sovereign Liquidity Lab",
    page_icon="🌐",
    layout="wide",
)


COUNTRY_URL = "https://api.worldbank.org/v2/country"
INDICATORS = {
    "FI.RES.TOTL.CD": "reserves_usd",
    "NE.IMP.GNFS.CD": "imports_usd",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "BN.CAB.XOKA.CD": "current_account_usd",
    "NY.GDP.MKTP.CD": "gdp_usd",
    "DT.DOD.DSTC.CD": "short_term_external_debt_usd",
}


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


def vulnerability_status(x):
    if pd.isna(x):
        return "n/a"
    if x < 30:
        return "low"
    if x < 60:
        return "moderate"
    return "high"


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


st.title("Sovereign Liquidity Lab")
st.caption(
    "Public-data monitoring of sovereign external liquidity vulnerability. "
    "Prototype only: not a credit rating, investment recommendation, or official assessment."
)

with st.spinner("Loading World Bank data and building global model..."):
    full_model, latest_model = build_model()

if full_model.empty or latest_model.empty:
    st.error(
        "The app could not retrieve enough data from the World Bank API. "
        "Please refresh the app in a minute. If the problem persists, the API may be temporarily unavailable."
    )
    st.stop()

st.sidebar.header("Filters")
regions = sorted(latest_model["region_name"].dropna().unique())
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)

statuses = ["low", "moderate", "high"]
selected_statuses = st.sidebar.multiselect(
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
col3.metric("High Vulnerability", f"{(filtered['vulnerability_status'] == 'high').sum():,.0f}")
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
        "low": "#2e7d32",
        "moderate": "#f9a825",
        "high": "#c62828",
        "n/a": "#9e9e9e",
    },
)
fig.update_layout(yaxis_title="", xaxis_title="Vulnerability score")
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
)
fig_region.update_layout(yaxis_title="", xaxis_title="Average vulnerability score")
st.plotly_chart(fig_region, use_container_width=True)

st.subheader("Country Monitor")
country_names = ranking["name"].sort_values().tolist()
selected_country_name = st.selectbox("Select country", country_names)
country_row = ranking[ranking["name"] == selected_country_name].iloc[0]

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
