from __future__ import annotations

###############################################################################
#                                IMPORTS
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# ─── project‑specific helpers ────────────────────────────────────────────────
from streamlit_app import (
    load_and_preprocess_data,     # returns (allocations, quarterly_returns)
    load_individual_endowments,   # returns annual endowment dataframe
)

def sharpe_ratio(returns: pd.Series, ann_factor: int = 1) -> float:
    """
    Annualised Sharpe ratio of a series of *excess* returns.

    Parameters
    ----------
    returns     : pd.Series – excess returns (decimal) at the native frequency
    ann_factor  : int       – periods per year (4 = quarterly, 12 = monthly, 1 = annual)
    """
    if returns.empty:
        return np.nan
    mu   = returns.mean()
    std  = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan
    return (mu / std) * np.sqrt(ann_factor)

def lo_adjusted_sharpe(returns: pd.Series,
                       ann_factor: int,
                       q: int | None = None) -> float:
    """
    Lo (2002) autocorrelation-adjusted Sharpe ratio.

    Parameters
    ----------
    returns     : pd.Series – excess returns (decimal) at the native frequency
    ann_factor  : int       – periods per year
    q           : int       – #lags for the adjustment (default = ann_factor)
    """
    if returns.empty:
        return np.nan
    if q is None:
        q = ann_factor

    # per-period Sharpe (not annualised yet)
    sr_per = sharpe_ratio(returns, ann_factor=1)

    # γ̂  = 1 + 2 Σ ρ_k (1 − k/(q+1))
    gamma_hat = 1.0
    for k in range(1, q + 1):
        rho_k = returns.autocorr(lag=k)
        if np.isnan(rho_k):
            continue
        gamma_hat += 2.0 * rho_k * (1.0 - k / (q + 1))

    if gamma_hat <= 0 or np.isnan(gamma_hat):
        return np.nan

    return sr_per / np.sqrt(gamma_hat) * np.sqrt(ann_factor)


def quarterly_to_annual(
    quarterly_returns: pd.Series,
    annual_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Aggregate quarterly returns into fiscal-year returns that align with the
    dates already used for the annual endowment series (usually 1 July).

    For each date *t* in *annual_index* we compound all quarterly observations
    in the half-open interval [t, next_t).

    Returns a Series whose index equals *annual_index*.
    """
    ann_ret = []
    for i, start in enumerate(annual_index):
        end = annual_index[i + 1] if i + 1 < len(annual_index) else start + pd.DateOffset(years=1)
        slice_q = quarterly_returns.loc[(quarterly_returns.index >= start) &
                                        (quarterly_returns.index <  end)]
        ann_ret.append((1.0 + slice_q).prod() - 1.0 if not slice_q.empty else np.nan)

    return pd.Series(ann_ret, index=annual_index, name=quarterly_returns.name)


def calculate_synthetic_endowment_returns(
    allocations: pd.DataFrame,
    returns_q: pd.DataFrame
) -> pd.Series:
    """
    Compute quarterly returns for the Synthetic Endowment Index.

    Steps
    -----
    1.  Map each SAA weight onto the quarterly return index.
    2.  Restrict to asset classes available in the return matrix.
    3.  Re-normalise the weights each quarter so they sum to 1.0.
    4.  Form the value-weighted return.

    Returns
    -------
    pd.Series – SEI quarterly returns (decimal), index identical to *returns_q*.
    """
    w_q = map_allocations_to_periods(allocations, returns_q.index)
    common = w_q.columns.intersection(returns_q.columns)
    w_q, r_q = w_q[common], returns_q[common]

    # renormalise weights each period, then compute the dot product row-wise
    w_norm = w_q.div(w_q.sum(axis=1), axis=0).fillna(0.0)
    sei = (w_norm * r_q).sum(axis=1)
    sei.name = "Synthetic Endowment Index"
    return sei


def map_allocations_to_periods(
    allocations_df: pd.DataFrame,
    date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Expand the historical strategic‐asset‐allocation (SAA) so that every date
    in *date_index* carries the correct asset-class weight.

    Parameters
    ----------
    allocations_df : long-format DataFrame exactly as returned by
        `load_and_preprocess_data()`
        (columns: Year, Asset Class, Allocation, Start Date, End Date).
    date_index     : target DatetimeIndex, usually the quarterly return index.

    Returns
    -------
    DataFrame of weights (decimals; rows = date_index, cols = asset classes).
    """
    alloc = pd.DataFrame(index=date_index).sort_index()

    for asset in allocations_df["Asset Class"].unique():
        alloc[asset] = 0.0                              # initialise column
        rows = allocations_df[allocations_df["Asset Class"] == asset]
        for _, r in rows.iterrows():
            mask = (date_index >= r["Start Date"]) & (date_index <= r["End Date"])
            w = r["Allocation"]
            # convert 0-100 → 0-1 if necessary
            if w > 1.5:
                w = w / 100.0
            alloc.loc[mask, asset] = w

    return alloc


###############################################################################
#                     TIME‑PERIOD INTERSECTION (STRICT)
###############################################################################
###############################################################################
#  TIME-PERIOD INTERSECTION (STRICT)   ��� fixed to drop all-zero baseline rows
###############################################################################
def get_longest_shared_period_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Return the slice where *all* series overlap (non-NaN),
    excluding any artificial all-zero baseline rows."""
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return df

    # ── find common overlap ─────────────────────────────────────────────
    starts, ends = [], []
    for col in df.columns:
        s = df[col].dropna()
        if not s.empty:
            starts.append(s.index.min())
            ends.append(s.index.max())

    if not starts or not ends:
        return pd.DataFrame()

    max_start, min_end = max(starts), min(ends)
    if max_start > min_end:
        return pd.DataFrame()

    df_sub = df.loc[max_start:min_end].copy()
    df_sub.dropna(axis=1, how="all", inplace=True)

    # ── NEW: drop rows that are exactly zero across every column ────────
    all_zero = (df_sub.abs() < 1e-12).all(axis=1)
    df_sub = df_sub.loc[~all_zero]

    return df_sub



###############################################################################
#                DESCRIPTIVE STATS (+ annualisation)
###############################################################################
def compute_descriptive_stats(
    df: pd.DataFrame,
    *,
    rf: pd.Series | None = None,          # NEW – risk-free series (same freq.)
    ann_factor: int = 1,                  # 4 = quarterly, 12 = monthly, 1 = annual
    include_ar1: bool = False,
    include_sr: bool = True,              # NEW – toggle Sharpe columns
    q_lags: int | None = None,            # NEW – #lags for Lo (default = ann_factor)
) -> pd.DataFrame:
    """
    Return a DataFrame with Obs, Mean %, StdDev %, Sharpe, Lo-adj. Sharpe,
    Skewness, Excess Kurtosis and (optionally) AR(1).

    *Mean/StdDev* use raw returns; Sharpe columns use *excess* returns.

    Parameters
    ----------
    df            : DataFrame of decimal returns.
    rf            : Series of risk-free returns (same frequency).  If None,
                    Sharpe columns are skipped unless *include_sr=False*.
    ann_factor    : annualisation factor for mean/std/Sharpes.
    include_ar1   : whether to append the AR(1) column.
    include_sr    : whether to append Sharpe columns.
    q_lags        : #lags for the Lo adjustment (default = ann_factor).
    """
    df = df.dropna(axis=1, how="all")
    stats_dict: dict[str, dict[str, float]] = {}

    for col in df.columns:
        s_raw = df[col].dropna()
        if s_raw.empty:
            continue

        # ---------- basic moments (raw returns) --------------------------
        mean_ann = (1 + s_raw.mean()) ** ann_factor - 1
        std_ann  = s_raw.std(ddof=0) * np.sqrt(ann_factor)  # population sd
        skew     = stats.skew(s_raw, bias=False)
        ex_kurt  = stats.kurtosis(s_raw, bias=False)
        n_obs    = len(s_raw)

        row = {
            "Obs": n_obs,
            "Mean (%)": mean_ann * 100,
            "StdDev (%)": std_ann * 100,
            "Skewness": skew,
            "Excess Kurtosis": ex_kurt,
        }

        # ---------- Sharpe ratios (excess returns) -----------------------
        if include_sr and rf is not None:
            s_excess = s_raw.sub(rf.reindex_like(s_raw), fill_value=np.nan).dropna()
            if not s_excess.empty:
                row["Sharpe"]         = sharpe_ratio(s_excess, ann_factor)
                row["Sharpe (Lo)"]    = lo_adjusted_sharpe(s_excess,
                                                            ann_factor,
                                                            q_lags or ann_factor)

        if include_ar1:
            row["AR(1)"] = s_raw.autocorr(lag=1)

        stats_dict[col] = row

    # ---------- column order -------------------------------------------
    order = ["Obs", "Mean (%)", "StdDev (%)"]
    if include_sr and rf is not None:
        order += ["Sharpe", "Sharpe (Lo)"]
    order += ["Skewness", "Excess Kurtosis"]
    if include_ar1:
        order.append("AR(1)")

    return pd.DataFrame(stats_dict).T[order]



def add_obs_and_ar1(df: pd.DataFrame) -> pd.DataFrame:
    """Shortcut for monthly hedging‑strategy returns (1999‑07 to 2024‑06).

    Assumes *df* already truncated to that range and in decimal form.
    """
    return compute_descriptive_stats(df, ann_factor=12, include_ar1=True)

###############################################################################
#                         LaTeX HELPERS
###############################################################################
def latex_escape_percent(x: float) -> str:
    return f"{x:.2f}\\%"

def df_to_latex_rows(df: pd.DataFrame) -> list[str]:
    rows = []
    for idx, row in df.iterrows():
        vals = [
            latex_escape_percent(v) if "(%)" in col else f"{v:.2f}"
            for col, v in row.items()
        ]
        rows.append(f"{idx} & " + " & ".join(vals) + r" \\")
    return rows

def convert_panels_to_latex(
    df_q: pd.DataFrame,
    df_a: pd.DataFrame,
    caption: str,
    label: str,
    note: str,
) -> str:
    """
    Build a *single* LaTeX table with two panels (Quarterly & Annual).
    Mean / StdDev already annualised upstream.
    """
    note = note.replace("%", "\\%")  # escape percent
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\begin{tiny}",
        rf"\caption{{\normalsize{{{caption}}}\\ \footnotesize{{{note}}}}}",
        rf"\label{{{label}}}",
    ]

    # ─── Panel A (Quarterly) ──────────────────────────────────────────────
    lines += [
        r"\textbf{Panel A. Quarterly Asset‑Class Returns (annualised)}\par\smallskip",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc}",
        r"\toprule",
        " & " + " & ".join(df_q.columns) + r" \\",
        r"\midrule",
        *df_to_latex_rows(df_q),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{1em}",
    ]

    # ─── Panel B (Annual) ────────────────────────────────────────────────
    lines += [
        r"\textbf{Panel B. Annual Endowment Returns}\par\smallskip",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc}",
        r"\toprule",
        " & " + " & ".join(df_a.columns) + r" \\",
        r"\midrule",
        *df_to_latex_rows(df_a),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{tiny}",
        r"\end{table}",
    ]
    return "\n".join(lines)


###############################################################################
#            LOAD  HEDGING STRATEGIES  (MONTHLY)  – unchanged
###############################################################################
def load_hedging_strategies_for_stats() -> pd.DataFrame:
    fp = "data/hedging_strategies.csv"
    df = pd.read_csv(fp, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df.set_index("Date", inplace=True)
    df.index = df.index + pd.offsets.MonthEnd(0)

    for col in df.columns:
        df[col] = (
            df[col]
            .str.rstrip("%")
            .replace("", np.nan)
            .astype(float)
            .div(100)
        )
    df.sort_index(inplace=True)
    df.rename(
        columns={
            "V Fast": "TS Momentum (Very Fast)",
            "Fast": "TS Momentum (Fast)",
            "Med": "TS Momentum (Medium)",
            "Slow": "TS Momentum (Slow)",
            "V Slow": "TS Momentum (Very Slow)",
        },
        inplace=True,
    )

 # Add the TSM Basket as an equal-weight average of all speeds
    df["TS Momentum (Basket)"] = df[
        [
            "TS Momentum (Very Fast)",
            "TS Momentum (Fast)",
            "TS Momentum (Medium)",
            "TS Momentum (Slow)",
            "TS Momentum (Very Slow)",
        ]
    ].mean(axis=1)

    return df

def truncate_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc["1999-07-01":"2024-06-30"].copy()

###############################################################################
#                                 MAIN APP
###############################################################################
def main() -> None:
    """Streamlit dashboard: Table 1 descriptive stats incl. Sharpe & Lo-adj. Sharpe."""
    st.title("Descriptive Statistics Dashboard")

    # ─── 1  Load raw data (quarterly) ─────────────────────────────────────────
    allocations, ret_q_raw = load_and_preprocess_data()     # quarterly returns
    endow_ann  = load_individual_endowments()               # annual returns

    # ─── 2  Risk-free series  (Cash = USGG3M)  ───────────────────────────────
    rf_q = ret_q_raw["Cash"].copy()                         # quarterly --> decimal

    # ─── 3  Synthetic Endowment Index (needs full ret_q) ─────────────────────
    sei_q = calculate_synthetic_endowment_returns(allocations, ret_q_raw)
    ret_q_full = ret_q_raw.copy()
    ret_q_full["Synthetic Endowment Index"] = sei_q

    # ─── 4  Prepare annual endowment series ──────────────────────────────────
    sei_a = quarterly_to_annual(sei_q, endow_ann.index)
    endow_ann = endow_ann.copy()
    endow_ann["Synthetic Endowment Index"] = sei_a

    baseline_dt = pd.Timestamp("1999-07-01")
    if baseline_dt in endow_ann.index:
        endow_ann = endow_ann.drop(baseline_dt)

    # ─── 5  Drop Cash from the stats universe  ───────────────────────────────
    ret_q = ret_q_full.drop(columns=["Cash"])

    # ─── 6  Descriptive stats – Quarterly (annualised) ───────────────────────
    q_shared = get_longest_shared_period_strict(ret_q)
    if q_shared.empty:
        st.error("No overlapping quarterly range.")
        return

    desc_q = compute_descriptive_stats(
        q_shared,
        rf=rf_q,                 # EXCESS returns for Sharpe columns
        ann_factor=4,
        include_sr=True,
        q_lags=4                 # 4 quarterly lags in the Lo adjustment
    )

    # ─── 7  Descriptive stats – Annual  ──────────────────────────────────────
    rf_a = quarterly_to_annual(rf_q, endow_ann.index)       # align fiscal years
    a_shared = get_longest_shared_period_strict(endow_ann)
    if a_shared.empty:
        st.error("No overlapping annual range.")
        return

    desc_a = compute_descriptive_stats(
        a_shared,
        rf=rf_a,
        ann_factor=1,
        include_sr=True,
        q_lags=1
    )

    # ─── 8  Display Table 1 + LaTeX code ─────────────────────────────────────
    st.subheader("Table 1 – Descriptive Statistics (incl. Synthetic Endowment Index)")
    st.dataframe(
        pd.concat(
            {"Quarterly (ann.)": desc_q, "Annual": desc_a},
            axis=0,
            names=["Panel", "Series"]
        ).style.format("{:.2f}")
    )

    latex_tbl1 = convert_panels_to_latex(
        desc_q,
        desc_a,
        caption="Descriptive Statistics of Returns (Synthetic Endowment Index included)",
        label="tab:descriptive_stats",
        note=("Mean and StdDev are annualised where applicable; "
              "Sharpe columns use excess returns over Cash. "
              "Skewness and Excess Kurtosis are unit-less.")
    )
    st.code(latex_tbl1, language="latex")

    # ─── 9  Hedging-strategy appendix (Table A-1) ───────────────────────────
    hedging_df = truncate_period(load_hedging_strategies_for_stats())

    # monthly risk-free series: forward-fill the quarterly Cash returns
    rf_m = rf_q.resample("M").ffill().loc[hedging_df.index]

    hs_stats = compute_descriptive_stats(
        hedging_df,
        rf=rf_m,
        ann_factor=12,   # monthly → annual
        include_sr=True,
        q_lags=12        # Lo adjustment with 12 monthly lags
    )

    st.subheader("Hedging Strategy Descriptive Stats (Table A-1)")
    st.dataframe(hs_stats.style.format("{:.2f}"))

    st.code(
        convert_panels_to_latex(
            hs_stats, pd.DataFrame(),
            caption="Hedging Strategies – Descriptive Stats",
            label="table:hedging_stats",
            note=("Monthly data 1999-07-01 to 2024-06-30.  "
                "Mean & StdDev annualised; Sharpe ratios in excess of the "
                "3-month Treasury bill, Lo-adjusted with 12 monthly lags."),
        ),
        language="latex",
    )

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()


