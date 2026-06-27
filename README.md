# Sovereign Liquidity Lab

A public-data prototype for monitoring sovereign external liquidity vulnerability.

The tool combines reserve adequacy, short-term external debt coverage, current-account pressure, and external liquidity gap indicators into a transparent country-level monitoring framework.

## Current Prototype

The first Streamlit version includes:

- Global external vulnerability ranking
- Regional summary dashboard
- Country-level monitor
- External vulnerability score
- Baseline external liquidity gap estimate
- Downloadable CSV ranking

## Methodology

The prototype uses public World Bank data and calculates:

- Import cover: total reserves divided by monthly imports
- Reserves / short-term external debt
- Current account balance, percent of GDP
- Baseline liquidity gap: estimated external financing needs minus usable reserves after preserving a three-month import-cover floor

The composite score uses:

- 40% import-cover risk
- 40% reserves / short-term external debt risk
- 20% current-account risk

## Planned Modules

- Stress scenario lab
- FX market-pressure layer
- Country snapshot generator
- Reserve-risk and yield-curve monitor
- Methodology note and data-quality flags

## Disclaimer

This project is for analytical and educational purposes only. It is not a credit rating, investment recommendation, trading signal, or official assessment.
