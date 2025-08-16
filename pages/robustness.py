# pages/robustness.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import rcParams
from decimal import Decimal, ROUND_HALF_UP
from matplotlib.ticker import PercentFormatter

from utils import (
    load_asset_allocation_and_returns_data,
    map_allocations_to_periods,
    load_hedge_strategies,
    calculate_cumulative_returns_and_drawdowns,
    calculate_sei_returns,
    trend_sensitivity,
    detect_crisis_periods,
    protection_cost_ratio,
    cost_adjusted_risk_reduction,
    median_crisis_payoff,
)

# ------------------------------------------------------------------
# Debug switch (visible in Streamlit sidebar)
if "Debug" not in st.session_state:
    st.session_state["Debug"] = False
with st.sidebar:
    st.checkbox("Debug TSM numbers", key="Debug")
# ------------------------------------------------------------------


rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
})

# ───────────────────────────────────────────────────────── helpers ──
def annualised_from_growth(growth: float, days: int) -> float:
    """Annualise a total growth factor observed over *days* calendar days."""
    if days <= 0 or pd.isna(growth) or growth <= 0:
        return np.nan
    years = days / 365.25
    return growth ** (1 / years) - 1


def combined_crisis_cagr(cum_ser: pd.Series, crises: list[dict]) -> float:
    """
    Compound the growth factors across ALL crisis windows, then annualise
    over the total number of crisis days.  Non-crisis periods are ignored.
    """
    growth = 1.0
    total_days = 0
    for c in crises:
        s, e = c["Start"], c["End"]
        if s in cum_ser.index and e in cum_ser.index:
            growth *= cum_ser.loc[e] / cum_ser.loc[s]
            total_days += (e - s).days
    return annualised_from_growth(growth, total_days) if total_days else np.nan


def pct_1dig(x: float | None) -> str:
    """Format x (a proportion, e.g. 0.234) as 1‑digit % with ROUND_HALF_UP."""
    if x is None or pd.isna(x):
        return "NA"
    return f"{Decimal(x*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"

def peak_to_trough(cum_series: pd.Series, start: pd.Timestamp, trough: pd.Timestamp) -> float:
    """
    Raw (not annualized) return from start → trough:
      (cum_trough / cum_start) - 1
    """
    if start not in cum_series.index or trough not in cum_series.index:
        return np.nan
    return cum_series.loc[trough] / cum_series.loc[start] - 1


def unify_timeframe(alloc, ret):
    start = max(alloc["Start Date"].min(), ret.index.min())
    end = min(alloc["End Date"].max(), ret.index.max())
    alloc_f = alloc[(alloc["End Date"] >= start) & (alloc["Start Date"] <= end)]
    ret_f   = ret.loc[start:end]
    return alloc_f, ret_f, start, end


def quarters_diff(d1, d2):
    days = (d2 - d1).days
    return max(int(round(days / 91.3125)), 1)


def build_crisis_table(crises, cum_hedge):
    """
    Build a dataframe of peak‑to‑trough pay‑offs for each hedge overlay
    in every crisis window.

    * Hedge columns are returned as raw floats (or NaN) — no formatting here.
    * Summary rows (mean / median) are also stored as floats.
    """
    if not crises:
        return pd.DataFrame()

    hedge_keys = list(cum_hedge.keys())
    rows = []

    for idx, c in enumerate(crises, start=1):
        s, t, e, md = c["Start"], c["Trough"], c["End"], c["Max Drawdown"]
        row = {
            "#": idx,
            "Start":   s.date(),
            "Trough":  t.date(),
            "End":     e.date(),
            "Max DD":  f"{md:.1%}",
            "Depth(Q)": quarters_diff(s, t),
            "Recov(Q)": quarters_diff(t, e),
        }
        # --- store raw float pay‑offs (no rounding yet) ---------------
        for h in hedge_keys:
            row[h] = peak_to_trough(cum_hedge[h], s, t)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- summary helper (mean / median) -----------------------
    def summary(stat, label):
        out = {c: "" for c in df.columns}
        out["#"], out["Max DD"] = label, f"--{label}--"
        for h in hedge_keys:
            v = getattr(df[h].dropna().astype(float), stat)()
            out[h] = v
        return out

    df = pd.concat(
        [df,
         pd.DataFrame([summary("mean", "ALL AVG"),
                       summary("median", "ALL MED")])],
        ignore_index=True
    )
    return df


def to_latex(df, title, threshold):
    align = "l" + "c"*(df.shape[1]-1)
    caption = (
        rf"\caption{{\normalsize{{{title}}}\\" +
        rf"\footnotesize{{Drawdown threshold: {threshold*100:.1f}\%}}}}"
    )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\begin{tiny}",
        caption,
        rf"\label{{table:{title.lower().replace(' ','_')}}}",
        rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{align}}}",
        r"\toprule"
    ]
    cols = df.columns.tolist()
    lines.append(" & ".join([""]+cols) + r" \\")
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


def generate_crisis_table(name: str,
                          crises: list[dict],
                          cum_hedge: dict[str, pd.Series],
                          threshold: float) -> None:
    """Display crisis table with % numbers rounded to one decimal."""

    df = build_crisis_table(crises, cum_hedge)
    if df.empty:
        st.info(f"No crises for {name}.");  return

    hedge_cols = list(cum_hedge.keys())

    # build a format‑mapping:  one‑decimal % for every hedge column
    fmt_map = {c: (lambda x, _=c: pct_1dig(x)) for c in hedge_cols}

    # Depth/Recov = integers, leave as default; Max DD already a string
    st.subheader(f"{name} — {int(threshold*100)}% Threshold")
    st.dataframe(df.style.format(fmt_map), height=300)

    st.code(to_latex(df, f"{name} Crises {int(threshold*100)}%", threshold),
            language="latex")


# ───────────────────────────────────────────────────────── generate_crisis_plot ──
def generate_crisis_plot(crises, cum_hedge, trend_sens, threshold):
    """
    Plot median, mean, and combined-crisis CAGR for PuE & SEI
    across the five TSM signal speeds.  Key names in cum_hedge use the
    full descriptive labels, so we map the numerical look-back lengths
    to those column names here.
    """
    st.write(f"### {int(threshold*100)}% Threshold")

    # full column names in the hedging DataFrame
    speed2col = {
        4:  "Time Series Momentum (Very Fast)",
        7:  "Time Series Momentum (Fast)",
        12: "Time Series Momentum (Med)",
        20: "Time Series Momentum (Slow)",
        24: "Time Series Momentum (Very Slow)",
    }

    # ---------- helpers --------------------------------------------------
    def pt_stats(ser, cr_list):
        vals = [ser.loc[c["Trough"]] / ser.loc[c["Start"]] - 1
                for c in cr_list
                if c["Start"] in ser.index and c["Trough"] in ser.index]
        return (np.nanmedian(vals) if vals else np.nan,
                np.nanmean(vals)   if vals else np.nan)

    def ann_combined(ser, cr_list):
        return combined_crisis_cagr(ser, cr_list)

    # ---------- build series --------------------------------------------
    pe_med, pe_mean, pe_ann = ([] for _ in range(3))
    sei_med, sei_mean, sei_ann = ([] for _ in range(3))

    for sp in trend_sens:
        col = speed2col.get(sp)
        if col is None or col not in cum_hedge:
            # skip if the specified sleeve is missing
            continue

        ser = cum_hedge[col]

        m_pu, a_pu  = pt_stats(ser, crises["Public Equity"])
        m_se, a_se  = pt_stats(ser, crises["Synthetic Endowment"])

        pe_med.append(m_pu);  pe_mean.append(a_pu)
        sei_med.append(m_se); sei_mean.append(a_se)

        pe_ann.append(ann_combined(ser, crises["Public Equity"]))
        sei_ann.append(ann_combined(ser, crises["Synthetic Endowment"]))

    # ---------- plotting -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colours = {"PuE": "black", "SEI": "#6e6e6e"}   # brighter gray for SEI

    x_vals = [sp for sp in trend_sens if speed2col.get(sp) in cum_hedge]

    # PuE lines
    l1, = ax.plot(x_vals, pe_med,  "o-", label="PuE Median P-T", color=colours["PuE"])
    l2, = ax.plot(x_vals, pe_mean, "s--", label="PuE Mean P-T",  color=colours["PuE"])
    l3, = ax.plot(x_vals, pe_ann,  "^:", label="PuE Crisis CAGR",color=colours["PuE"])

    # SEI lines
    l4, = ax.plot(x_vals, sei_med,  "o-", label="SEI Median P-T", color=colours["SEI"])
    l5, = ax.plot(x_vals, sei_mean, "s--", label="SEI Mean P-T",  color=colours["SEI"])
    l6, = ax.plot(x_vals, sei_ann,  "v:", label="SEI Crisis CAGR",color=colours["SEI"])

    ymax = np.nanmax(pe_med + pe_mean + pe_ann + sei_med + sei_mean + sei_ann)
    ax.set_ylim(0, ymax * 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Trend Sensitivity (weeks)")
    ax.set_ylabel("Return")
    ax.set_xticks(x_vals)
    ax.grid(axis="y", linestyle="--", lw=0.5, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend([l1, l2, l3, l4, l5, l6],
              [ln.get_label() for ln in (l1, l2, l3, l4, l5, l6)],
              ncol=3, frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(pad=0.5)
    st.pyplot(fig)


def process_threshold(th, cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens):
    pe_cr = detect_crisis_periods(dd_pe, th)
    se_cr = detect_crisis_periods(dd_sei, th)

    generate_crisis_table("Public Equity", pe_cr, cum_hedge, th)
    generate_crisis_table("Synthetic Endowment", se_cr, cum_hedge, th)
    generate_crisis_plot({"Public Equity":pe_cr, "Synthetic Endowment":se_cr},
                         cum_hedge, trend_sens, th)


def process_top_crises(cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens, n=4):
    top_pe  = sorted(detect_crisis_periods(dd_pe), key=lambda c: abs(c["Max Drawdown"]), reverse=True)[:n]
    top_sei = sorted(detect_crisis_periods(dd_sei),key=lambda c: abs(c["Max Drawdown"]), reverse=True)[:n]
    st.header(f"Top {n} Largest Crises")
    generate_crisis_table("Public Equity", top_pe, cum_hedge, 0.0)
    generate_crisis_table("Synthetic Endowment", top_sei, cum_hedge, 0.0)
    generate_crisis_plot({"Public Equity":top_pe, "Synthetic Endowment":top_sei},
                         cum_hedge, trend_sens, 0.0)
    
###############################################################################
#          FAST vs SLOW TSM — 10 % CRISIS TABLE  (LEVEL‑BASED PAY‑OFF)
###############################################################################
def debug_single_hedge(cum: pd.Series, crises: list[dict], label: str):
    """Print raw peak‑to‑trough pay‑offs so we can compare with LaTeX table."""
    rows = []
    for c in crises:
        if c["Start"] in cum.index and c["Trough"] in cum.index:
            payoff = cum.loc[c["Trough"]] / cum.loc[c["Start"]] - 1
            rows.append({
                "Start":   c["Start"].date(),
                "Trough":  c["Trough"].date(),
                "Payoff":  f"{payoff:.1%}"
            })
    if rows:
        st.write(f"#### Debug – raw pay‑offs for {label}")
        st.table(pd.DataFrame(rows))
        st.write("Median =", pd.Series(
            [float(r["Payoff"].rstrip('%')) / 100 for r in rows]).median().round(4))

###############################################################################
#      FAST vs SLOW TSM — Table with RELATIVE IMPROVEMENTS
#      (carry = geometric CAGR of MONTHLY returns  ← original version)
###############################################################################
def tsm_speed_table(idx_q: pd.DatetimeIndex,
                    pu_ret: pd.Series,
                    sei_ret: pd.Series,
                    hedge_monthly: pd.DataFrame,
                    threshold: float = 0.10) -> None:

    # 1 ─ detect crises
    _, dd_pu  = calculate_cumulative_returns_and_drawdowns(pu_ret)
    _, dd_sei = calculate_cumulative_returns_and_drawdowns(sei_ret)
    pu_cr  = detect_crisis_periods(dd_pu,  threshold)
    sei_cr = detect_crisis_periods(dd_sei, threshold)

    # 2 ─ cumulative TSM indices on quarter-ends
    tags = {"V Fast", "Fast", "Med", "Slow", "V Slow"}
    cum_q = {}
    for col in hedge_monthly:
        if ("momentum" in col.lower()) or (col.strip() in tags):
            cum = (1 + hedge_monthly[col].dropna()).cumprod()
            cum_q[col] = cum.asfreq("QE-DEC", method="ffill").reindex(idx_q)

    if not cum_q:
        st.warning("No TSM data found.");  return

    # 3 ─ keep only crises fully covered by every overlay
    def covered(cr):
        s, e = cr["Start"], cr["End"]
        return all(s in srs.index and e in srs.index for srs in cum_q.values())

    pu_cr  = [c for c in pu_cr  if covered(c)]
    sei_cr = [c for c in sei_cr if covered(c)]

    rows = []
    for name, cum_qtr in cum_q.items():
        cum_qtr = cum_qtr.dropna()
        if cum_qtr.empty:
            continue

        # ---------- peak-to-trough pay-offs --------------------------
        pt = lambda ser, cr: ser.loc[cr["Trough"]] / ser.loc[cr["Start"]] - 1
        pu_vals  = [pt(cum_qtr, c) for c in pu_cr]
        sei_vals = [pt(cum_qtr, c) for c in sei_cr]

        med_pu,  med_sei  = map(np.nanmedian, (pu_vals, sei_vals))
        mean_pu, mean_sei = map(np.nanmean,  (pu_vals, sei_vals))

        # ---------- combined crisis CAGR (new) -----------------------
        ann_pu  = combined_crisis_cagr(cum_qtr, pu_cr)
        ann_sei = combined_crisis_cagr(cum_qtr, sei_cr)

        # ---------- long-run carry ----------------------------------
        ret_m = hedge_monthly[name].dropna()
        carry = (1 + ret_m).prod() ** (1 / (len(ret_m) / 12)) - 1 if len(ret_m) else np.nan

        rows.append({
            "Strategy":  name,
            "Median PuE":  med_pu,   "Median SEI":  med_sei,
            "Mean PuE":    mean_pu,  "Mean SEI":    mean_sei,
            "Crisis CAGR PuE %":  ann_pu  * 100,
            "Crisis CAGR SEI %":  ann_sei * 100,
            "Annualised Carry %": carry   * 100,
        })

    if not rows:
        st.info("No crises with full TSM coverage."); return

    df = pd.DataFrame(rows).set_index("Strategy")

    pct1 = lambda x: "NA" if pd.isna(x) else f"{x*100:.1f}%"
    one  = "{:.1f}".format
    fmt  = {"Median PuE": pct1, "Median SEI": pct1,
            "Mean PuE":   pct1, "Mean SEI":   pct1,
            "Crisis CAGR PuE %": one, "Crisis CAGR SEI %": one,
            "Annualised Carry %": one}

    st.subheader("Fast vs Slow TSM (10 % drawdown)")
    st.dataframe(df.style.format(fmt), height=460)


def main():
    st.title("Robustness Checks — thresholds, largest drawdowns & TSM")

    alloc, ret_q = load_asset_allocation_and_returns_data()
    alloc_f, ret_f, start, end = unify_timeframe(alloc, ret_q)

    idx_q = pd.date_range(start, end, freq="QE-DEC")
    sei_ret = calculate_sei_returns(alloc_f, ret_f).reindex(idx_q).dropna()
    pe_ret  = ret_f["Public Equity"].reindex(idx_q).dropna()

    cum_sei, dd_sei = calculate_cumulative_returns_and_drawdowns(sei_ret)
    cum_pe,  dd_pe  = calculate_cumulative_returns_and_drawdowns(pe_ret)

    hedge_df = load_hedge_strategies()
    cum_hedge = {
        name: calculate_cumulative_returns_and_drawdowns(ser.dropna())[0]
        for name, ser in hedge_df.items() if ser.dropna().size>1
    }

    trend_sens = trend_sensitivity()

    for thr in [0.10, 0.05, 0.15]:      # 10 %, 5 %, 15 %
        st.header(f"{int(thr*100)}% Threshold")
        process_threshold(thr, cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens)

    process_top_crises(cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens, n=4)

    # ------------------ TSM speed table ------------------------------
    tsm_speed_table(idx_q, pe_ret, sei_ret, hedge_df, 0.10)


if __name__ == "__main__":
    main()
