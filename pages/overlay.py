import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List

from utils import (
    load_asset_allocation_and_returns_data,
    calculate_sei_returns,
    calculate_cumulative_returns_and_drawdowns,
    load_hedge_strategies,
    detect_crisis_periods,
)

###############################################################################
#                               ‑‑ helpers ‑‑
###############################################################################

def pct_1dig(x: float | None) -> str:
    """Format ratio as one‑decimal percent (ROUND_HALF_UP)."""
    if x is None or pd.isna(x):
        return "NA"
    return f"{Decimal(x*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def annualise_growth(growth: float, days: int) -> float:
    if days <= 0 or pd.isna(growth) or growth <= 0:
        return np.nan
    years = days / 365.25
    return growth ** (1 / years) - 1


def crisis_cagr(ret_m: pd.Series, crises: list[dict]) -> float:
    if ret_m.empty or not crises:
        return np.nan

    growth, total_days = 1.0, 0
    for cr in crises:
        s, e = cr["Start"], cr["End"]
        seg = ret_m.loc[s:e]
        if seg.empty:
            continue
        growth *= (1 + seg).prod()
        total_days += (e - s).days
    return annualise_growth(growth, total_days)


def ann_return(ret: pd.Series, periods_per_year: int = 12) -> float:
    if ret.empty:
        return np.nan
    yrs = len(ret) / periods_per_year
    return (1 + ret).prod() ** (1 / yrs) - 1


def payoff_stats(ret_m: pd.Series, crises: list[dict]) -> tuple[float, float]:
    if ret_m.empty or not crises:
        return np.nan, np.nan
    cum = (1 + ret_m).cumprod().reindex(
        pd.date_range(ret_m.index.min(), ret_m.index.max(), freq="D")
    ).ffill()
    vals = []
    for cr in crises:
        s, t = cr["Start"], cr["Trough"]
        if s in cum.index and t in cum.index:
            vals.append(cum.loc[t] / cum.loc[s] - 1)
    return (np.nanmedian(vals), np.nanmean(vals)) if vals else (np.nan, np.nan)


def crisis_ann_return(ret_m: pd.Series, crises: list[dict]) -> float:
    if ret_m.empty or not crises:
        return np.nan
    vals = []
    for cr in crises:
        seg = ret_m.loc[cr["Start"]:cr["End"]]
        vals.append(ann_return(seg))
    return np.nanmedian(vals) if vals else np.nan


def build_table(title: str,
                crises: list[dict],
                hedges: Dict[str, pd.Series]) -> None:
    rows: list[dict] = []

    for name, ret_m in hedges.items():
        ret_m = ret_m.dropna()
        if ret_m.empty:
            continue

        med_pt, mean_pt = payoff_stats(ret_m, crises)
        crisis_ann = crisis_cagr(ret_m, crises)
        uncond_ann = ann_return(ret_m)

        rows.append({
            "Strategy": name,
            "Median": med_pt,
            "Mean": mean_pt,
            "Crisis Ann. Return %": crisis_ann * 100,
            "Uncond. Ann. Return %": uncond_ann * 100,
        })

    if not rows:
        st.warning(f"No returns available for {title.lower()} window(s).")
        return

    df = pd.DataFrame(rows).set_index("Strategy")

    fmt = {"Median": pct_1dig,
           "Mean": pct_1dig,
           "Crisis Ann. Return %": "{:.1f}".format,
           "Uncond. Ann. Return %": "{:.2f}".format}

    st.subheader(title)
    st.dataframe(
        df.style.format(fmt)
          .set_caption(title),
        height=min(420, 40 + 24*len(df))
    )

###############################################################################
#                       ‑‑ helper for overlay returns ‑‑
###############################################################################

def monthly_sei_returns(alloc_df: pd.DataFrame, returns_q: pd.DataFrame) -> pd.Series:
    """Convert quarterly SEI returns to month‑end frequency."""
    sei_q = calculate_sei_returns(alloc_df, returns_q).dropna()
    if sei_q.empty:
        return pd.Series(dtype=float)
    # cumulative at quarter‑ends
    sei_cum_q = (1 + sei_q).cumprod()
    m_idx = pd.date_range(sei_q.index.min(), sei_q.index.max(), freq="M")
    sei_cum_m = sei_cum_q.reindex(m_idx).ffill()
    sei_ret_m = sei_cum_m.pct_change().dropna()
    sei_ret_m.name = "SEI"
    return sei_ret_m


def compute_overlay(sei_m: pd.Series, hedge_m: pd.Series, w_hedge: float = 0.10) -> pd.Series:
    """Return monthly 90/10 overlay between SEI and hedge (weights fixed)."""
    joined = pd.concat([sei_m, hedge_m], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    overlay = (1 - w_hedge) * joined.iloc[:, 0] + w_hedge * joined.iloc[:, 1]
    overlay.name = f"SEI 90/10 {hedge_m.name}"
    return overlay

###############################################################################
#                            ‑‑ Streamlit App ‑‑
###############################################################################

def main():
    st.title("Synthetic Endowment 90/10 Overlay — Crisis Tables")

    # ---------- load data ----------
    alloc, ret_q = load_asset_allocation_and_returns_data()
    sei_m = monthly_sei_returns(alloc, ret_q)

    if sei_m.empty:
        st.error("Unable to compute monthly SEI returns — check input data.")
        return

    hedge_df = load_hedge_strategies()
    # Align timeframe
    hedge_df = hedge_df.loc[sei_m.index.min():sei_m.index.max()]

    # Build overlay return series
    hedges_ret: Dict[str, pd.Series] = {"SEI (100%)": sei_m}
    for col in hedge_df.columns:
        h = hedge_df[col].dropna()
        if h.empty:
            continue
        overlay = compute_overlay(sei_m, h, w_hedge=0.10)
        if overlay.empty:
            continue
        hedges_ret[f"SEI 90/10 {col}"] = overlay

    # ---------- crisis detection on SEI & Public Equity ----------
    sei_cum, sei_dd = calculate_cumulative_returns_and_drawdowns(
        calculate_sei_returns(alloc, ret_q)
    )
    pe_series = ret_q["Public Equity"].dropna()
    pe_cum, pe_dd = calculate_cumulative_returns_and_drawdowns(pe_series)

    # ---------- numeric thresholds (default 10%) ----------
    threshold = st.sidebar.slider("Draw‑down Threshold (%)", 5, 20, 10, 1) / 100

    sei_crises = detect_crisis_periods(sei_dd, threshold)
    pe_crises = detect_crisis_periods(pe_dd, threshold)

    st.header(f"{int(threshold*100)} % Draw‑down Threshold")
    build_table(f"SEI‑defined Crises ({int(threshold*100)} %)", sei_crises, hedges_ret)
    build_table(f"Public Equity‑defined Crises ({int(threshold*100)} %)", pe_crises, hedges_ret)


if __name__ == "__main__":
    main()
