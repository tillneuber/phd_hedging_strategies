# pages/crisis_tsm_table.py
"""
Trend-Speed Momentum ─ Crisis Tables (10 % Threshold)
─────────────────────────────────────────────────────
• Two tables are produced:
      1. Crises defined on the Synthetic Endowment Index (SEI)
      2. Crises defined on Public Equity (PuE)
• Columns
      – Median / Mean peak-to-trough pay-off (%)
      – Crisis Ann. Return %   (median across windows)
      – Uncond. Ann. Return %  (sample-wide)
"""

import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict

from utils import (
    load_asset_allocation_and_returns_data,
    load_hedge_strategies,
    calculate_cumulative_returns_and_drawdowns,
    calculate_sei_returns,
    detect_crisis_periods,
)

# -----------------------------------------------------------------
def get_top_crises(crises: List[Dict], n: int = 4) -> List[Dict]:
    """
    Keep the |n| largest (absolute) peak-to-trough draw-downs from an
    already-detected `crises` list.
    """
    return sorted(crises, key=lambda c: abs(c["Max Drawdown"]),
                  reverse=True)[:n]

# ──────────────────────────────────────────────────── NEW helper ──
def load_tsm_hedges() -> dict[str, pd.Series]:
    """
    Return a dictionary with only the five individual TSM sleeves
    plus the equal-weighted TSM Basket, each as a clean monthly return
    series (NaNs dropped).
    """
    hedge_monthly = load_hedge_strategies()
    tsm_dict = {
        name: ser.dropna()
        for name, ser in hedge_monthly.items()
        if "Time Series Momentum" in name
    }
    return tsm_dict

# -----------------------------------------------------------------
def process_threshold(
        title_prefix: str,
        threshold: float,
        sei_cum, sei_dd,
        pe_cum,  pe_dd,
        hedges: Dict[str, pd.Series]) -> None:
    """
    Build a pair of tables (SEI-defined crises & PuE-defined crises)
    for a single numeric threshold.
    """
    sei_crises = detect_crisis_periods(sei_dd, threshold)
    pe_crises  = detect_crisis_periods(pe_dd,  threshold)

    build_table(f"{title_prefix} (SEI, {int(threshold*100)}%)",
                sei_crises, hedges)
    build_table(f"{title_prefix} (PuE, {int(threshold*100)}%)",
                pe_crises,  hedges)


# -----------------------------------------------------------------
def process_top_n(title_prefix: str,
                  n: int,
                  sei_dd, pe_dd,
                  hedges: Dict[str, pd.Series]) -> None:
    """
    Build a pair of tables for the |n| deepest draw-downs, irrespective
    of any explicit percentage threshold.
    """
    sei_crises = get_top_crises(detect_crisis_periods(sei_dd), n)
    pe_crises  = get_top_crises(detect_crisis_periods(pe_dd),  n)

    build_table(f"{title_prefix} (SEI Top {n})", sei_crises, hedges)
    build_table(f"{title_prefix} (PuE Top {n})", pe_crises,  hedges)


# ──────────────────────────────────────────────────── helpers ──
def pct_1dig(x: float | None) -> str:
    """Format ratio as one-decimal percent (ROUND_HALF_UP)."""
    if x is None or pd.isna(x):
        return "NA"
    return f"{Decimal(x*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"

# ──────────────────────────────────────────────────── helpers ──
def annualise_growth(growth: float, days: int) -> float:
    """Convert a total growth factor observed over *days* into a CAGR."""
    if days <= 0 or pd.isna(growth) or growth <= 0:
        return np.nan
    years = days / 365.25
    return growth ** (1 / years) - 1


def crisis_cagr(ret_m: pd.Series, crises: list[dict]) -> float:
    """
    Annualised return computed by:
      • keeping only returns inside crisis windows,
      • compounding them across ALL crises,
      • annualising over the TOTAL number of crisis days.
    """
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
    """Annualised geometric mean of simple returns."""
    if ret.empty:
        return np.nan
    yrs = len(ret) / periods_per_year
    return (1 + ret).prod()**(1 / yrs) - 1

def unify_timeframe(alloc: pd.DataFrame, ret: pd.DataFrame):
    """Trim allocation & return data to their common intersection."""
    start = max(alloc["Start Date"].min(), ret.index.min())
    end   = min(alloc["End Date"].max(),  ret.index.max())
    alloc = alloc[(alloc["End Date"] >= start) & (alloc["Start Date"] <= end)]
    ret   = ret.loc[start:end]
    return alloc, ret, start, end

def payoff_stats(ret_m: pd.Series, crises: list[dict]) -> tuple[float, float]:
    """Median & mean peak-to-trough pay-offs across given crises."""
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
    """Median annualised hedge return from start→recovery across crises."""
    if ret_m.empty or not crises:
        return np.nan
    vals = []
    for cr in crises:
        seg = ret_m.loc[cr["Start"]:cr["End"]]
        vals.append(ann_return(seg))
    return np.nanmedian(vals) if vals else np.nan

def build_table(title: str,
                crises: list[dict],
                hedges: dict[str, pd.Series]) -> None:
    """
    Assemble and display a Streamlit table for one crisis definition
    (SEI-defined or PuE-defined).

    Columns
    -------
    Median / Mean           : peak-to-trough pay-offs (percentage)
    Crisis Ann. Return %    : CAGR over the concatenated crisis windows
                              (all non-crisis months discarded)
    Uncond. Ann. Return %   : whole-sample CAGR

    Notes
    -----
    * `pct_1dig` formats proportions as x.x %.
    * `crisis_cagr` is the new helper that compounds returns only inside
      crisis windows, then annualises over the total number of crisis days.
    """
    rows: list[dict] = []

    for name, ret_m in hedges.items():
        ret_m = ret_m.dropna()
        if ret_m.empty:
            continue

        # peak-to-trough statistics
        med_pt, mean_pt = payoff_stats(ret_m, crises)

        # NEW: combined crisis CAGR
        crisis_ann = crisis_cagr(ret_m, crises)

        # unconditional CAGR
        uncond_ann = ann_return(ret_m)

        rows.append({
            "Strategy":              name,
            "Median":                med_pt,
            "Mean":                  mean_pt,
            "Crisis Ann. Return %":  crisis_ann * 100,
            "Uncond. Ann. Return %": uncond_ann * 100,
        })

    if not rows:
        st.warning(f"No hedge overlays cover the {title.lower()} windows.")
        return

    df = (pd.DataFrame(rows)
            .set_index("Strategy")
            .sort_index())

    # formatting
    fmt = {"Median": pct_1dig,
           "Mean":   pct_1dig,
           "Crisis Ann. Return %": "{:.1f}".format,   # one-decimal
           "Uncond. Ann. Return %": "{:.2f}".format}

    st.subheader(title)
    st.dataframe(
        df.style.format(fmt)
          .set_caption(f"{title} (10 % threshold)"),
        height=min(420, 40 + 24*len(df))
    )


# ──────────────────────────────────────────────────── page ──
st.title("Trend-Speed Momentum — Crisis Tables")

# ---------- load data (unchanged) -----------------------------
alloc, ret_q = load_asset_allocation_and_returns_data()
alloc, ret_q, start, end = unify_timeframe(alloc, ret_q)

idx_q   = pd.date_range(start, end, freq="QE-DEC")
sei_ret = calculate_sei_returns(alloc, ret_q).reindex(idx_q).dropna()
pe_ret  = ret_q["Public Equity"].reindex(idx_q).dropna()

sei_cum, sei_dd = calculate_cumulative_returns_and_drawdowns(sei_ret)
pe_cum,  pe_dd  = calculate_cumulative_returns_and_drawdowns(pe_ret)

hedge_monthly  = load_hedge_strategies()
hedges_ret = load_tsm_hedges()

    # ---------- numeric thresholds -------------------------------
for thr in (0.05, 0.10, 0.15):
    st.header(f"{int(thr*100)} % Draw-down Threshold")
    process_threshold("Crisis Tables", thr,
        sei_cum, sei_dd,
        pe_cum,  pe_dd,
        hedges_ret)

# ---------- Top-4 deepest draw-downs --------------------------
st.header("Four Largest Peak-to-Trough Declines")
process_top_n("Largest-DD Tables", 4,
        sei_dd, pe_dd,
        hedges_ret)