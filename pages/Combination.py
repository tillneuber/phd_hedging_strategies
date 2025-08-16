# ────────────────────────────────────────────────────────
#  pages/overlay_90_10_crises.py
#
#  90 % Synthetic Endowment Index (SEI) +
#  10 % Hedge-strategy overlays, re-balanced quarterly.
#
#  • Crisis windows are defined on the SEI itself
#    (default: –10 % peak-to-trough draw-down).
#  • Table shows raw peak-to-trough pay-offs for:
#        – 100 % SEI benchmark
#        – every 90/10 individual overlay
#        – NEW 90/10 “TSM Basket” (2 % each in the five trend speeds)
#  • ALL-AVG and ALL-MED summary rows included.
#  • Copy-ready LaTeX code is produced.
# ────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import scipy.stats as stats   # already available in Streamlit’s runtime



from utils import (                       # repo-shipped helpers
    load_asset_allocation_and_returns_data,
    load_hedge_strategies,
    calculate_sei_returns,
    calculate_cumulative_returns_and_drawdowns,
    detect_crisis_periods,
)

# ───────────────────────────── parameters ──
SEI_WEIGHT       = 0.90           # 90 / 10 overlay
CRISIS_THRESHOLD = 0.10           # 10 % draw-down ⇒ crisis
TABLE_HEIGHT     = 360

# ───────────────────────────── helper fns ──
def pct_1dig(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{Decimal(x*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"

def qret_or_nan(s):
    """Quarterly return; NaN if the entire quarter is missing."""
    return (1 + s).prod() - 1 if s.notna().any() else np.nan

def quarters_diff(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    return max(int(round((d2 - d1).days / 91.3125)), 1)

# ─────────────────── NEW helper functions ─────────────────────────
def sharpe_ratio(
    r: pd.Series,
    rf: pd.Series | None = None,
    ann_factor: int = 4
) -> float:
    """
    Annualised Sharpe ratio of *excess* returns.
    If *rf* is None the function treats *r* as already excess returns.
    """
    if rf is not None:
        r = r.sub(rf.reindex_like(r), fill_value=np.nan)
    r = r.dropna()
    if r.empty or r.std(ddof=1) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(ann_factor)


def lo_adjusted_sharpe(
    r: pd.Series,
    rf: pd.Series | None = None,
    ann_factor: int = 4,
    q: int = 4
) -> float:
    """
    Lo-adjusted Sharpe ratio (Lo, 2002) with *q* lags.
    """
    if rf is not None:
        r = r.sub(rf.reindex_like(r), fill_value=np.nan)
    r = r.dropna()
    if r.empty or r.std(ddof=1) == 0:
        return np.nan

    sr_per = r.mean() / r.std(ddof=1)              # per-period SR
    gamma  = 1.0 + 2.0 * sum(
        r.autocorr(lag=k) * (1 - k / (q + 1)) for k in range(1, q + 1)
    )
    if gamma <= 0 or np.isnan(gamma):
        return np.nan
    return sr_per / np.sqrt(gamma) * np.sqrt(ann_factor)


# ─────────────────── UPDATED (replace)  moment_table_annualised ───────────────────
def moment_table_annualised(
    ret_series_dict: dict[str, pd.Series],
    *,
    rf_q: pd.Series | None = None,   # <— add risk-free series
    ann_factor: int = 4,
    q_lags: int = 4
) -> pd.DataFrame:
    """
    Annualised unconditional moments plus Sharpe ratios.

    Columns:
        Obs, Mean, StdDev, SR, SR_Lo, Skewness, ExcessKurtosis
    """
    rows = []
    for lbl, r in ret_series_dict.items():
        r = r.dropna()
        if r.empty:
            continue

        mean_ann = (1.0 + r.mean()) ** ann_factor - 1.0
        std_ann  = r.std(ddof=1) * np.sqrt(ann_factor)

        rows.append(dict(
            Series         = lbl,
            Obs            = len(r),
            Mean           = mean_ann * 100,
            StdDev         = std_ann  * 100,
            SR             = sharpe_ratio(r, rf_q, ann_factor),
            SR_Lo          = lo_adjusted_sharpe(r, rf_q, ann_factor, q_lags),
            Skewness       = r.skew(),
            ExcessKurtosis = r.kurtosis(),      # already excess
        ))

    ordered = ["Obs", "Mean", "StdDev", "SR", "SR_Lo", "Skewness", "ExcessKurtosis"]
    return pd.DataFrame(rows).set_index("Series")[ordered]


# ───────────────────────────────────────────────────────── helpers ──
def peak_to_trough(cum: pd.Series,
                   start: pd.Timestamp,
                   trough: pd.Timestamp) -> float:
    """
    Raw peak-to-trough return for a cumulative-level series.

    The pay-off is only computed if *both* `start` **and** `trough`
    exist in the index **and** hold non-NaN values.  
    Otherwise the function returns `np.nan`, signalling that this
    crisis window is not covered by the overlay.

    Parameters
    ----------
    cum    : pd.Series
        Cumulative level path (starts at 1.0).
    start  : pd.Timestamp
        Crisis start date.
    trough : pd.Timestamp
        Crisis trough date.

    Returns
    -------
    float
        (cum_trough / cum_start) − 1, or np.nan if the window is
        not fully covered.
    """
    if start not in cum.index or trough not in cum.index:
        return np.nan

    v_start, v_trough = cum.loc[start], cum.loc[trough]
    if pd.isna(v_start) or pd.isna(v_trough):
        return np.nan

    return v_trough / v_start - 1


def _style_for_series(label: str) -> dict:
    """
    Journal-of-Finance friendly palette.

    • SEI (most important)      → solid black
    • 90/10 TSM Basket          → solid mid-grey
    • Public Equity (reference) → dash-dot dark grey
    • Any extra overlays        → light grey dashed
    """
    if label == "100 % SEI":                       # primary
        return dict(color="black", linestyle="solid",   linewidth=1.4, zorder=3)

    if "TSM Basket" in label:                      # secondary, still solid
        return dict(color="#6E6E6E", linestyle="solid", linewidth=1.3, zorder=2)

    if label == "Public Equity":                   # reference, dash-dot
        return dict(color="#4D4D4D", linestyle="dashdot", linewidth=1.3, zorder=2)

    # fallback for any additional overlays
    return dict(color="#A0A0A0", linestyle="dashed", linewidth=1.1, zorder=1)


# ───────────────────────────────────────────────────────────────
def performance_and_dd_chart(
    cum_map: dict[str, pd.Series],
    plot_order: list[str],
    crises: list[dict] | None = None,
) -> plt.Figure:
    """
    Two-panel chart (cumulative & draw-down).

    *Series are truncated at the earliest last-observation date across
     *all* requested lines, so everyone ends together.*
    """
    legend_name = {
        "100 % SEI": "SEI",
        "Public Equity": "Public Equity",
        "90/10 TSM Basket": "90% SEI + 10% TSM Basket",
    }

    # ------------------------------------------------------------
    # Find the last quarter with *valid Trend-Signal data*
    # ------------------------------------------------------------
    # 1) try an explicit basket column …
    basket_cols = [c for c in hedge_q_full.columns            # <-- only "basket"
               if "basket" in c.lower()]

    if basket_cols:
        last_trend = hedge_q_full[basket_cols[0]].dropna().index.max()
    else:
        # 2) … otherwise take the 5 individual speeds
        speed_toks = ["vfast", "fast", "med", "slow", "vslow"]
        speed_cols = [c for c in hedge_q_full.columns
                    if any(tok in c.lower() for tok in speed_toks)]

        # dropna(how="all") keeps a row only if at least one speed is still alive
        last_trend = hedge_q_full[speed_cols].dropna(how="all").index.max()

    # Use *that* as the hedge sample limit
    common_end = min(sei_ret_q_full.dropna().index.max(), last_trend)

    if DEBUG:
        st.sidebar.write(f"common_end set to {common_end.date()} (trend data end)")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 4.6))

    # crisis shading
    if crises and common_end is not None:
        for cr in crises:
            if cr["Start"] <= common_end:
                ax1.axvspan(cr["Start"], min(cr["End"], common_end),
                            color="lightgray", alpha=0.3, zorder=0)
                ax2.axvspan(cr["Start"], min(cr["End"], common_end),
                            color="lightgray", alpha=0.3, zorder=0)

    # plot loops
    for lab in plot_order:
        s_full = cum_map.get(lab, pd.Series(dtype=float)).dropna()
        if s_full.empty:
            continue
        s = s_full.loc[:common_end] if common_end is not None else s_full

        ax1.plot(s.index, s.values,
                 label=legend_name.get(lab, lab),
                 **_style_for_series(lab))

        dd = (s / s.cummax()) - 1
        ax2.plot(dd.index, dd.values,
                 label="_nolegend_", **_style_for_series(lab))

    # cosmetics
    ax1.set_ylabel("Cumulative Returns", fontsize=9)
    ax1.set_ylim(bottom=0.0)
    ax2.set_ylabel("Drawdowns", fontsize=9)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), ncol=len(handles),
               frameon=False, fontsize=8)

    fig.tight_layout(pad=1.2)
    return fig


# ───────────────────────────────────────────────────────── helpers ──
def annualise_growth(growth: float, days: int) -> float:
    """Annualise a total growth factor observed over *days* calendar days."""
    if days <= 0 or pd.isna(growth) or growth <= 0:
        return np.nan
    years = days / 365.25
    return growth ** (1 / years) - 1


def precise_crisis_cagr(cum_ser: pd.Series,
                        crises: list[dict]) -> float:
    """
    Annualised CAGR from exact crisis start→end levels.
    Windows are used only if *both* dates exist *and* hold non-NaN
    values in the cumulative series.
    """
    if cum_ser.empty or not crises:
        return np.nan

    growth, covered_days = 1.0, 0
    for cr in crises:
        s, e = cr["Start"], cr["End"]
        if (s in cum_ser.index and e in cum_ser.index and
            pd.notna(cum_ser.loc[s]) and pd.notna(cum_ser.loc[e])):

            growth *= cum_ser.loc[e] / cum_ser.loc[s]
            covered_days += (e - s).days

    return annualise_growth(growth, covered_days) if covered_days else np.nan




# ───────────────────────────────────────────────────────────────
def build_crisis_table(crises: list[dict],
                       cum_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Assemble a crisis-by-crisis pay-off table **plus** summary rows.

    * For every hedge (and the 100 % SEI reference) a pay-off is
      calculated with `peak_to_trough()`.  
      If the overlay does **not** cover the full window the entry
      is `np.nan`, shows up as *NA* in Streamlit, and is excluded
      from the mean/median rows.

    * Two extra rows are appended:
        – “Uncond Ann Return”  whole-sample CAGR  
        – “Crisis Ann Return”  CAGR restricted to crisis windows
    """
    if not crises:
        return pd.DataFrame()

    hedge_cols = list(cum_dict.keys())
    rows = []

    # ── 1) one row per crisis ───────────────────────────────────────
    for idx, cr in enumerate(crises, start=1):
        s, t, e, md = cr["Start"], cr["Trough"], cr["End"], cr["Max Drawdown"]
        row = {
            "#": idx,
            "Start":   s.date(),
            "Trough":  t.date(),
            "End":     e.date(),
            "Max DD":  f"{md:.1%}",
            "Depth(Q)": quarters_diff(s, t),
            "Recov(Q)": quarters_diff(t, e),
        }
        for h in hedge_cols:
            ser = cum_dict[h]
            # —— require all three dates to exist *and* be non-NaN ——
            if (s in ser.index and t in ser.index and e in ser.index and
                pd.notna(ser.loc[s]) and pd.notna(ser.loc[t]) and pd.notna(ser.loc[e])):
                row[h] = ser.loc[t] / ser.loc[s] - 1
            else:
                row[h] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 2) summary helpers (mean / median) ──────────────────────────
    def summary(stat: str, label: str):
        out = {c: "" for c in df.columns}
        out["#"], out["Max DD"] = label, f"--{label}--"
        for h in hedge_cols:
            out[h] = getattr(df[h].dropna().astype(float), stat)()
        return out

    df = pd.concat(
        [df,
         pd.DataFrame([summary("mean", "ALL AVG"),
                       summary("median", "ALL MED")])],
        ignore_index=True
    )

    # ── 3) unconditional & crisis CAGRs (unchanged) ────────────────
    uncond = {c: "" for c in df.columns}
    crisis = {c: "" for c in df.columns}
    uncond["#"], crisis["#"] = "Uncond Ann Return", "Crisis Ann Return"
    uncond["Max DD"], crisis["Max DD"] = "--UR--", "--CR--"

    for h in hedge_cols:
        cum = cum_dict[h]
        q_rets = cum.pct_change().dropna()
        yrs = len(q_rets) / 4
        uncond[h] = ((1 + q_rets).prod()) ** (1 / yrs) - 1 if yrs else np.nan
        crisis[h] = precise_crisis_cagr(cum, crises)

    df = pd.concat([df, pd.DataFrame([uncond, crisis])], ignore_index=True)
    return df

def to_latex(df: pd.DataFrame, caption: str, threshold: float) -> str:
    align = "l" + "c" * (df.shape[1] - 1)
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\begin{tiny}",
        rf"\caption{{\normalsize{{{caption}}}\\"
        rf"\footnotesize{{Drawdown threshold: {threshold*100:.1f}\%}}}}",
        rf"\label{{table:{caption.lower().replace(' ','_')}}}",
        rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{align}}}",
        r"\toprule"
    ]
    cols = df.columns.tolist()
    lines.append(" & ".join([""] + cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("%", r"\%") for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{tiny}",
        r"\end{table}"
    ]
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────
def build_cum_dict(
    idx_q: pd.DatetimeIndex,
    sei_ret_q: pd.Series,
    cum_sei: pd.Series,
    hedge_q: pd.DataFrame,
    sei_w: float = 0.90,debug=False,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """
    Create parallel dictionaries:

        • cum_dict[label]  – cumulative level series (starts at 1.0)
        • ret_dict[label]  – quarterly simple-return series (NaN where
                              the overlay cannot be formed)

    Naming convention
    -----------------
    The basket overlay is *always* stored under the key
        "90/10 TSM Basket"
    no matter how the hedge file names the underlying column(s).
    """

    # ---- helper functions -----------------------------------------
    def _norm(s: str) -> str:
        """lower-case and strip non-alphanumerics – for fuzzy matching."""
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _embed(part: pd.Series) -> pd.Series:
        """Embed a partial series into the full quarterly index."""
        out = pd.Series(index=idx_q, dtype=float)
        out.loc[part.index] = part
        return out

    # ----------------------------------------------------------------
    cum_dict: dict[str, pd.Series] = {"100 % SEI": cum_sei}
    ret_dict: dict[str, pd.Series] = {"100 % SEI": sei_ret_q}

    # ----------------------------------------------------------------
    # 1)  Individual 90/10 overlays (one per hedge column)
    # ----------------------------------------------------------------
    for col in hedge_q.columns:
        h = hedge_q[col].reindex(idx_q)            # keep NaNs
        if h.notna().sum() == 0:                   # never trades
            continue

        valid = h.notna()
        r_ov = pd.Series(np.nan, index=idx_q)
        r_ov.loc[valid] = sei_w * sei_ret_q.loc[valid] + (1 - sei_w) * h.loc[valid]

        # Use standard basket label if the column itself is a basket
        std_label = (
            "90/10 TSM Basket"
            if "basket" in _norm(col) and "timeseriesmomentum" in _norm(col)
            else f"90/10 {col}"
        )

        # Avoid duplicates if we later build a basket from the five speeds
        if std_label in ret_dict:
            continue

        ret_dict[std_label] = r_ov
        cum_dict[std_label] = _embed(
            calculate_cumulative_returns_and_drawdowns(r_ov.dropna())[0]
        )

    # ----------------------------------------------------------------
    # 2)  Build an equal-weight basket from the *five* trend speeds,
    #     if they exist and we have not already added the basket.
    # ----------------------------------------------------------------
    if "90/10 TSM Basket" not in ret_dict:
        # fuzzy-match to find one column for each speed
        token_map = {
            "vfast": None, "fast": None, "med": None, "slow": None, "vslow": None
        }
        for col in hedge_q.columns:
            n = _norm(col)
            for tok in token_map:
                if tok in n and token_map[tok] is None:
                    token_map[tok] = col
                    break

        if all(token_map.values()):                # found all five speeds
            panel = hedge_q[list(token_map.values())].reindex(idx_q)
            good = panel.notna().all(axis=1)       # every speed present

            if good.any():
                basket_ret = (0.02 * panel[good]).sum(axis=1)
                combo_ret = pd.Series(np.nan, index=idx_q)
                combo_ret.loc[good] = (
                    sei_w * sei_ret_q.loc[good] + basket_ret
                )

                ret_dict["90/10 TSM Basket"] = combo_ret
                cum_dict["90/10 TSM Basket"] = _embed(
                    calculate_cumulative_returns_and_drawdowns(combo_ret.dropna())[0]
                )

    return cum_dict, ret_dict


# ───────────────────────────────────────────────────────────────
def nav_spending_series(ret_q: pd.Series,
                        initial_nav: float = 100.0,
                        payout_rate: float = 0.05
                        ) -> pd.DataFrame:
    """
    Build a quarterly NAV path given *ret_q* (simple returns).

    For each quarter t:
        1. NAV_pre = NAV_post_{t-1} × (1 + r_t)
        2. Spend_t = (payout_rate / 4) × mean( NAV_pre_t plus 11 prior NAV_post )
        3. NAV_post_t = NAV_pre − Spend_t
        4. Spend DD_t = (Spend_t − max_{τ≤t} Spend_τ) / max_{τ≤t} Spend_τ

    Returns
    -------
    DataFrame indexed like *ret_q* with columns
        ["NAV", "Spending", "Spend DD"]   (Spend DD in decimal form, 0 = no drawdown)
    """
    nav_vals, spend_vals, dd_vals = [], [], []
    trailing_pre = []
    nav_prev_post = initial_nav
    run_max_spend = 0.0

    for r in ret_q:
        nav_pre = nav_prev_post * (1 + r)

        trailing_pre.append(nav_pre)
        if len(trailing_pre) > 12:
            trailing_pre.pop(0)

        base   = sum(trailing_pre) / len(trailing_pre)
        spend  = (payout_rate / 4) * base
        nav_post = nav_pre - spend

        run_max_spend = max(run_max_spend, spend)
        spend_dd = 0.0 if run_max_spend == 0 else (spend - run_max_spend) / run_max_spend

        nav_vals.append(nav_post)
        spend_vals.append(spend)
        dd_vals.append(spend_dd)

        nav_prev_post = nav_post

    return pd.DataFrame({"NAV": nav_vals,
                         "Spending": spend_vals,
                         "Spend DD": dd_vals},
                        index=ret_q.index)


# ───────────────────────────────────────────────────────────────
def build_nav_spending_table(
    ret_dict: dict[str, pd.Series],
    initial_nav: float = 100.0,
    payout_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Assemble NAV / Spending / Spend-DD panel.

    • Quarters with NaN returns are excluded before the NAV simulation,
      preventing NaNs from poisoning the whole path.
    """
    frames = []
    for label, r in ret_dict.items():
        r_clean = r.dropna()
        if r_clean.empty:
            continue

        nav_df = nav_spending_series(r_clean, initial_nav, payout_rate)
        nav_df.columns = pd.MultiIndex.from_product([[label], nav_df.columns])
        frames.append(nav_df)

    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def _end_weights_from_returns(
    start_w: pd.Series,
    r_q:     pd.Series
) -> pd.Series:
    """
    End-of-quarter weights *before* rebalancing, given
    target start-weights **start_w** (sum to 1.0) and realised
    quarterly simple returns **r_q** for the same asset list.

    w_endᵢ = w_startᵢ·(1+rᵢ)  /  Σⱼ w_startⱼ·(1+rⱼ)
    """
    growth  = (1 + r_q).mul(start_w, fill_value=np.nan)
    return growth / growth.sum()


# ─────────────────── REPLACE THE OLD VERSION WITH THIS ────────────
def calculate_quarterly_weight_drift(
    alloc_q:     pd.DataFrame,   # long format: one row per sleeve-period
    ret_q:       pd.DataFrame,   # wide format: sleeves in columns, quarterly idx
    hedge_ret_q: pd.Series,      # Trend-Basket quarterly returns
    sei_weight:  float = 0.90,
) -> pd.DataFrame:
    """
    Absolute weight drift |w_end − w_target| for every SEI sleeve **plus**
    a single “Hedge” column (the 10 % block).  Values are in **% of NAV**.

    * alloc_q columns required:  "Asset Class", "Allocation",
                                 "Start Date",  "End Date"
    * ret_q columns must contain the same sleeve names as *Asset Class*.
    * ret_q and hedge_ret_q must share the same quarterly DatetimeIndex.
    """
    # ── 1) universe of sleeves available in BOTH tables ────────────
    sleeves = sorted(
        set(alloc_q["Asset Class"].unique())
        .intersection(ret_q.columns)
    )
    if not sleeves:
        raise ValueError(
            "No common sleeve names between alloc_q['Asset Class'] "
            "and ret_q.columns.  Check spelling & capitalisation."
        )

    idx_q = ret_q.index

    # ── 2) build target-weight panel (wide) from the long table ────
    w_target = pd.DataFrame(np.nan, index=idx_q, columns=sleeves)
    for _, row in alloc_q.iterrows():
        asset = row["Asset Class"]
        if asset not in sleeves:
            continue
        mask = (idx_q >= row["Start Date"]) & (idx_q <= row["End Date"])
        w_target.loc[mask, asset] = row["Allocation"]

    w_target = w_target.mul(sei_weight)   # scale to 90 % block
    w_target["Hedge"] = 1.0 - sei_weight  # 10 % block

    # ── 3) returns panel aligned to the same columns ───────────────
    r_panel = ret_q.loc[idx_q, sleeves].copy()
    r_panel["Hedge"] = hedge_ret_q.reindex(idx_q)

    # ── 4) helper to go from start- to end-weights within a quarter
    def _end_weights(w_start: pd.Series, r_q: pd.Series) -> pd.Series:
        growth = (1 + r_q).mul(w_start, fill_value=np.nan)
        return growth / growth.sum()

    # ── 5) drift calculation loop ─────────────────────────────────
    records = []
    for t in idx_q:
        w_start = w_target.loc[t].dropna()
        r_q     = r_panel.loc[t].dropna()
        common  = w_start.index.intersection(r_q.index)
        if common.empty:
            records.append(pd.Series(name=t, dtype=float))
            continue
        w_end  = _end_weights(w_start[common], r_q[common])
        records.append((w_end - w_start[common]).abs())

    return pd.concat(records, axis=1).T.reindex(idx_q)

# ─────────────────── NEW helper: dedicated drift chart ────────────
def plot_weight_drift(
    drift_df: pd.DataFrame,
    crises:    list[dict] | None = None,
    figsize:   tuple = (8.0, 3.4),
) -> plt.Figure:
    """
    Time-series plot of quarterly absolute weight-drift (% of NAV)
    with *distinct colours* for every sleeve + Hedge block.
    """
    if drift_df.empty:
        raise ValueError("drift_df is empty.")

    # --- colour palette (tab10 → 10, falls back to tab20 beyond that) ----------
    n_series  = drift_df.shape[1]
    cmap_name = "tab10" if n_series <= 10 else "tab20"
    palette   = plt.cm.get_cmap(cmap_name, n_series)

    fig, ax = plt.subplots(figsize=figsize)

    # crisis shading
    if crises:
        last_date = drift_df.index.max()
        for cr in crises:
            if cr["Start"] <= last_date:
                ax.axvspan(cr["Start"], min(cr["End"], last_date),
                           color="lightgray", alpha=0.3, zorder=0)

    # plotting loop with distinct colours
    for i, col in enumerate(drift_df.columns):
        ax.plot(
            drift_df.index, drift_df[col],
            label=col,
            color=palette(i),              # distinct colour
            linewidth=1.2,
        )

    ax.set_ylabel("Absolute Drift (% of NAV)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout(pad=0.8)
    return fig


# ───────────────────────────────────────────────────────────────
def plot_spending_dd(
    nav_table: pd.DataFrame,
    series_order: list[str] | None = None,
    crises: list[dict] | None = None,
) -> plt.Figure:
    """
    Spending draw-down chart.

    *The x-axis is truncated to the earliest last-date shared by
     **all** plotted overlays.*
    """
    available = set(nav_table.columns.get_level_values(0))
    if series_order is None:
        series_order = sorted(available)
    else:
        series_order = [lbl for lbl in series_order if lbl in available]
    if not series_order:
        raise ValueError("Requested labels not found in NAV table.")

    # common end-date ------------------------------------------------
    last_dates = [
        nav_table[(lbl, "Spend DD")].dropna().index.max() for lbl in series_order
    ]
    common_end = min(last_dates)

    legend_name = {
        "100 % SEI": "SEI",
        "90/10 TSM Basket": "90% SEI + 10% TSM Basket",
        "Public Equity": "Public Equity",
    }

    fig, ax = plt.subplots(figsize=(7.8, 3.8))

    # crisis shading
    if crises:
        for cr in crises:
            if cr["Start"] <= common_end:
                ax.axvspan(cr["Start"], min(cr["End"], common_end),
                           color="lightgray", alpha=0.3, zorder=0)

    # plotting loop
    handles = []
    for lbl in series_order:
        dd = nav_table[(lbl, "Spend DD")].dropna().loc[:common_end]
        if dd.empty:
            continue
        ln, = ax.plot(dd.index, dd.values,
                      label=legend_name.get(lbl, lbl),
                      **_style_for_series(lbl))
        handles.append(ln)

    ax.set_ylabel("Spending Drawdown", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=len(handles), frameon=False, fontsize=8)

    fig.tight_layout(pad=0.6)
    return fig

# ─────────────────── NEW helper: extended drift summary ───────────
def drift_extended_summary_table(
    drift_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Mean, Median, 25th/75th pct, 95th pct, Max, plus row 'Total'
    that sums across sleeves for every stat.
    """
    base = pd.DataFrame({
        "Mean":    drift_df.mean(),
        "Median":  drift_df.median(),
        "P25":     drift_df.quantile(0.25),
        "P75":     drift_df.quantile(0.75),
        "P95":     drift_df.quantile(0.95),
        "Max":     drift_df.max(),
    }).T

    base.loc["Total"] = base.sum(axis=1)
    return base

def common_sample_end(sei_q: pd.Series, hedge_q: pd.DataFrame) -> pd.Timestamp:
    """
    Last quarter that is still covered by
        • SEI returns, and
        • the *Trend basket* (or its five speeds).

    Anything that drifts on after the trend data ends is ignored.
    """
    last_sei = sei_q.dropna().index.max()

    # --- preferred: explicit “…basket…” column --------------------
    basket_cols = [c for c in hedge_q.columns if "basket" in c.lower()]
    if basket_cols:
        last_trend = hedge_q[basket_cols].dropna(how="all").index.max()
    else:
        # --- otherwise: take the five speed columns ---------------
        speed_toks = ["vfast", "fast", "med", "slow", "vslow"]
        speed_cols = [c for c in hedge_q.columns
                      if any(tok in c.lower() for tok in speed_toks)]
        last_trend = hedge_q[speed_cols].dropna(how="all").index.max()

    return min(last_sei, last_trend)



# ─────────────────── FINAL helper: uniform-style box-plot ──────────
def plot_drift_boxplot(
    drift_df: pd.DataFrame,
    figsize: tuple = (8.0, 3.4),
) -> plt.Figure:
    """
    Journal-style box-and-whisker plot of quarterly drift.

    • All boxes share the **same light-grey fill** – no per-sleeve colours.
    • Y-axis in percent; clean, grid-free axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        [drift_df[c].dropna() for c in drift_df.columns],
        labels=drift_df.columns,
        patch_artist=True,
        showfliers=False,
        medianprops = dict(color="black", linewidth=1.2),
        whiskerprops= dict(color="#4D4D4D", linewidth=1.0),
        capprops    = dict(color="#4D4D4D", linewidth=1.0),
    )

    # Uniform light-grey boxes
    for box in bp["boxes"]:
        box.set(facecolor="#D6D6D6", edgecolor="black", linewidth=1.0)

    ax.set_ylabel("Absolute Drift (% of NAV)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig



# ─────────────────── UPDATED helper: histogram of total turnover ──
def plot_total_drift_histogram(
    drift_df: pd.DataFrame,
    bins: int = 20,
    figsize: tuple = (8.0, 3.4),
) -> plt.Figure:
    """
    Histogram of Σ|drift| across all sleeves (per quarter).
    Y-axis shows the **percentage of quarters** (0–100 %), matching
    the percent formatting used elsewhere in the paper.
    """
    total = drift_df.sum(axis=1).dropna()
    weights = np.ones_like(total) / len(total)          # → proportions

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        total,
        bins=bins,
        weights=weights,
        edgecolor="black",
        color="#4D4D4D",
        alpha=0.85,
    )

    ax.set_xlabel("Total Turnover per Quarter (% of NAV)")
    ax.set_ylabel("Share of Quarters")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0–100 %
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


# ─────────────────── UPDATED helper: liquid vs illiquid hist ──────
def plot_liquid_vs_illiquid_hist(
    drift_df: pd.DataFrame,
    illiquid_cols: list[str],
    bins: int = 20,
    figsize: tuple = (8.0, 3.4),
) -> plt.Figure:
    """
    Histogram of quarterly turnover split into *Liquid* vs *Illiquid*
    sleeves.  Y-axis shows the **percentage of quarters.**
    """
    liquid_cols    = [c for c in drift_df.columns if c not in illiquid_cols]
    liquid_total   = drift_df[liquid_cols].sum(axis=1).dropna()
    illiquid_total = drift_df[illiquid_cols].sum(axis=1).dropna()

    w_liq   = np.ones_like(liquid_total)   / len(liquid_total)
    w_illq  = np.ones_like(illiquid_total) / len(illiquid_total)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        [liquid_total, illiquid_total],
        bins=bins,
        weights=[w_liq, w_illq],
        label=["Liquid", "Illiquid"],
        edgecolor="black",
        alpha=0.80,
        color=["black", "#6E6E6E"],
    )

    ax.set_xlabel("Turnover per Quarter (% of NAV)")
    ax.set_ylabel("Share of Quarters")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=0.6)
    return fig

# ───────────────────────────── Streamlit UI ──
st.sidebar.header("Controls")
DEBUG = st.sidebar.checkbox("Debug mode", value=False)   #  ← DEBUG is now defined
st.title("90 % SEI / 10 % Hedge Overlays — SEI Crisis Tables")

# ───────────────────────────── data prep ──
alloc_q, ret_q = load_asset_allocation_and_returns_data()

start = max(alloc_q["Start Date"].min(), ret_q.index.min())
end   = min(alloc_q["End Date"].max(),  ret_q.index.max())

# restrict to the overall window ------------------------------------------------
alloc_q = alloc_q[(alloc_q["End Date"] >= start) & (alloc_q["Start Date"] <= end)]
ret_q   = ret_q.loc[start:end]

# preliminary SEI returns (full window) ----------------------------------------
sei_ret_q_full = calculate_sei_returns(alloc_q, ret_q)

# hedge strategies  (monthly → quarterly) --------------------------------------
hedge_m_full = load_hedge_strategies()
hedge_q_full = hedge_m_full.resample("Q").apply(qret_or_nan)

common_end = common_sample_end(sei_ret_q_full, hedge_q_full)   # ≈ 2022-09-30
idx_q      = pd.date_range(start, common_end, freq="QE-DEC")
sei_ret_q  = sei_ret_q_full.reindex(idx_q).dropna()
hedge_q    = hedge_q_full.reindex(idx_q)

# now all downstream objects use the aligned index -----------------------------
cum_sei, dd_sei = calculate_cumulative_returns_and_drawdowns(sei_ret_q)

cum_dict, ret_dict = build_cum_dict(
    idx_q, sei_ret_q, cum_sei, hedge_q, SEI_WEIGHT, debug=DEBUG
)


# -------- Public Equity cumulative series ----------------------
pe_ret_q = ret_q["Public Equity"].reindex(idx_q).dropna()
cum_pe, _ = calculate_cumulative_returns_and_drawdowns(pe_ret_q)

# add PuE to the overlay dictionaries
cum_dict["Public Equity"] = cum_pe
ret_dict["Public Equity"] = pe_ret_q

# ───────────────────────────── crisis detection ──
sei_crises = detect_crisis_periods(dd_sei, CRISIS_THRESHOLD)

# ───────────────────────────── Streamlit UI ──
st.title("90 % SEI / 10 % Hedge Overlays — SEI Crisis Tables")

if not sei_crises:
    st.info("No SEI crises at the chosen threshold.")
    st.stop()

table = build_crisis_table(sei_crises, cum_dict)


fmt_map = {c: pct_1dig for c in cum_dict.keys()}
st.dataframe(table.style.format(fmt_map), height=TABLE_HEIGHT)

st.subheader("LaTeX code")
st.code(to_latex(table,
                 "SEI-defined crises — 90/10 overlays",
                 CRISIS_THRESHOLD),
        language="latex")

# ---------- NAV & Spending table ---------------------------------
nav_table = build_nav_spending_table(ret_dict)

if DEBUG:
    st.sidebar.subheader("Keys present after NAV table build")
    st.sidebar.write({
        "cum_dict": list(cum_dict.keys()),
        "ret_dict": list(ret_dict.keys()),
        "nav_table": sorted(set(nav_table.columns.get_level_values(0)))
    })

st.header("Quarterly NAV and Spending (5 % annual, 12-Q avg)")
st.dataframe(nav_table, height=480)

# ---------- Spending draw-down chart -----------------------------
st.subheader("Spending Draw-down Trajectories")
fig_spend_dd = plot_spending_dd(nav_table, crises=sei_crises)
st.pyplot(fig_spend_dd)

# ---------- Performance & draw-down chart -----------------------
perf_labels = ["100 % SEI", "Public Equity", "90/10 TSM Basket"]

cum_for_plot = {"100 % SEI": cum_sei,
                "Public Equity": cum_pe}
if "90/10 TSM Basket" in cum_dict:
    cum_for_plot["90/10 TSM Basket"] = cum_dict["90/10 TSM Basket"]

st.subheader("Cumulative Performance and Draw-downs")
fig_perf = performance_and_dd_chart(cum_for_plot, perf_labels, crises=sei_crises)
st.pyplot(fig_perf)

# ---------- Spending draw-down chart (SEI vs 90/10 TSM) ----------
st.subheader("Spending Draw-down — SEI vs 90/10 TSM Basket")
fig_spend_dd_focus = plot_spending_dd(
    nav_table,
    series_order=["100 % SEI", "90/10 TSM Basket"]   # PuE removed
)
st.pyplot(fig_spend_dd_focus)

stats_df = moment_table_annualised(ret_dict)

fmt = {
    "Mean": "{:.2f} %",
    "StdDev": "{:.2f} %",
    "Skewness": "{:.2f}",
    "ExcessKurtosis": "{:.2f}"
}

st.subheader("Annualised Return Moments (Quarterly Data)")
st.dataframe(stats_df.style.format(fmt), height=330)

# ---------------------------------------------------------------
#  Overlap-aligned 90/10 overlay moment table  (Tail-Risk dropped)
# ---------------------------------------------------------------
# 1) pick all 90/10 overlays except Tail Risk
overlay_labels = [
    lbl for lbl in ret_dict
    if lbl.startswith("90/10") and "Tail Risk" not in lbl
]

# 2) find common date window where *every* series has data
first_dates = [ret_dict[lbl].first_valid_index() for lbl in overlay_labels]
last_dates  = [ret_dict[lbl].last_valid_index()  for lbl in overlay_labels]

common_start = max(d for d in first_dates if d is not None)
common_end   = min(d for d in last_dates  if d is not None)

# 3) build a trimmed dictionary (overlays + 100 % SEI as reference)
aligned_dict = {
    lbl: ret_dict[lbl].loc[common_start:common_end]
    for lbl in overlay_labels
}
aligned_dict["100 % SEI"] = sei_ret_q.loc[common_start:common_end]
aligned_dict["Public Equity"]   = ret_dict["Public Equity"].loc[common_start:common_end]


# 4) compute moments and show table
aligned_stats = moment_table_annualised(aligned_dict)
st.subheader(
    f"Annualised Return Moments — 90 / 10 Overlays "
    f"(common sample {common_start.date()} to {common_end.date()})"
)
st.dataframe(aligned_stats.style.format(fmt), height=330)

# 1)  Quarterly returns of the hedge block (Trend-Basket)
trend_ret_q = hedge_q.filter(regex="(?i)basket", axis=1)      # try explicit basket column
if trend_ret_q.empty:                                         # otherwise rebuild from 5 speeds
    speed_cols = [c for c in hedge_q.columns
                  if any(tok in c.lower() for tok in ["vfast", "fast", "med", "slow", "vslow"])]
    trend_ret_q = (0.02 * hedge_q[speed_cols]).sum(axis=1)
else:
    trend_ret_q = trend_ret_q.iloc[:, 0]                      # first (only) matching column

ret_q_aligned = ret_q.reindex(idx_q)      # <─ keep only dates ≤ common_end

drift_df = calculate_quarterly_weight_drift(
    alloc_q, ret_q.reindex(idx_q), trend_ret_q, sei_weight=SEI_WEIGHT
).dropna(how="all")      # <-- drop the empty 2022 Q4–2024 rows


# 3)  Show table & *colourful* chart
st.header("Quarterly Weight Drift vs Targets (pre-rebalance)")
st.dataframe(drift_df.style.format(pct_1dig), height=TABLE_HEIGHT)

st.subheader("Weight-Drift — Time Series")
fig_drift = plot_weight_drift(drift_df, crises=sei_crises)
st.pyplot(fig_drift)

st.subheader("LaTeX code (Weight Drift)")
st.code(
    to_latex(
        drift_df.reset_index().rename(columns={"index": "Quarter"}),
        "Quarterly Weight Drift vs Targets",
        SEI_WEIGHT
    ),
    language="latex"
)

# ---------- Extended summary: sleeve-level  +  Total / Liquid / Illiquid ------
ILLIQ_COLS = ["PE/VC", "Private Credit", "Real Assets & ILBs"]
LIQ_COLS   = [c for c in drift_df.columns if c not in ILLIQ_COLS]   # 5 liquid sleeves

def _stats(s: pd.Series) -> pd.Series:
    """Return the 6 summary stats for one series of quarterly drifts."""
    return pd.Series({
        "Mean"   : s.mean(),
        "Median" : s.median(),
        "P25"    : s.quantile(0.25),
        "P75"    : s.quantile(0.75),
        "P95"    : s.quantile(0.95),
        "Max"    : s.max(),
    })

# (1) sleeve-level stats  ───────────────────────────────────────────────
sleeve_stats = drift_df.apply(_stats).T                                # (7×6)

# (2) totals – recomputed from totals, NOT summed column-wise  ──────────
total_stats    = _stats(drift_df.sum(axis=1))
liquid_stats   = _stats(drift_df[LIQ_COLS].sum(axis=1))
illiquid_stats = _stats(drift_df[ILLIQ_COLS].sum(axis=1))

# (3) combine and order  ────────────────────────────────────────────────
ext_summary_df = (
    pd.concat(
        [sleeve_stats,
         pd.DataFrame([total_stats, liquid_stats, illiquid_stats],
                      index=["Total", "Liquid", "Illiquid"])]
    )
    .rename_axis("Block")
    .reindex(list(drift_df.columns) + ["Total", "Liquid", "Illiquid"])
)

# (4) show in Streamlit  ────────────────────────────────────────────────
st.subheader("Typical Quarter — Extended Summary (properly aggregated)")
st.dataframe(ext_summary_df.style.format(pct_1dig), height=360)


# ---------- Box-plot ----------------------------------------------
st.subheader("Distribution of Quarterly Drift by Sleeve")
fig_box = plot_drift_boxplot(drift_df)
st.pyplot(fig_box)

# ---------- Histogram ---------------------------------------------
st.subheader("How Often Do We Trade This Much?")
fig_hist = plot_total_drift_histogram(drift_df)
st.pyplot(fig_hist)

# ---------- Liquid vs. Illiquid turnover histogram ----------------
ILLIQ_COLS = ["PE/VC", "Private Credit", "Real Assets & ILBs"]

st.subheader("Turnover Split: Liquid vs. Illiquid Asset Blocks")
fig_liq_hist = plot_liquid_vs_illiquid_hist(drift_df, ILLIQ_COLS)
st.pyplot(fig_liq_hist)
