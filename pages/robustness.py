# pages/robustness.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import rcParams
from decimal import Decimal, ROUND_HALF_UP

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
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
})



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


def generate_crisis_table(name, crises, cum_hedge, threshold):
    st.subheader(f"{name} — {int(threshold*100)}% Threshold")
    df = build_crisis_table(crises, cum_hedge)
    st.dataframe(df, height=300)
    st.code(to_latex(df, f"{name} Crises {int(threshold*100)}%", threshold), language="latex")


def generate_crisis_plot(crises, cum_hedge, trend_sens, threshold):
    st.write(f"### {int(threshold*100)}% Threshold")

    # map speeds to hedge‐column names
    speed2col = {4: "V Fast", 7: "Fast", 12: "Med", 20: "Slow", 24: "V Slow"}

    # build median+mean lists
    def stats_for(cr):
        med, mn = [], []
        for sp in trend_sens:
            col = speed2col[sp]
            vals = [peak_to_trough(cum_hedge[col], c["Start"], c["Trough"]) for c in cr]
            vals = [v for v in vals if pd.notna(v)]
            med.append(np.nanmedian(vals) if vals else np.nan)
            mn.append(np.nanmean(vals)   if vals else np.nan)
        return med, mn

    pe_med, pe_mean = stats_for(crises["Public Equity"])
    sei_med, sei_mean = stats_for(crises["Synthetic Endowment"])

    fig, ax = plt.subplots(figsize=(5,3.5))
    colors = {"Public Equity":"black", "Synthetic Endowment":"dimgray"}

    ax.plot(trend_sens, pe_med,   "o-", label="PuE Median",  color=colors["Public Equity"])
    ax.plot(trend_sens, pe_mean,  "s--",label="PuE Mean",    color=colors["Public Equity"])
    ax.plot(trend_sens, sei_med,  "o-", label="SEI Median",  color=colors["Synthetic Endowment"])
    ax.plot(trend_sens, sei_mean, "s--",label="SEI Mean",    color=colors["Synthetic Endowment"])

    ax.set_xlabel("Trend Sensitivity (weeks)")
    ax.set_ylabel("Peak–Trough Return")
    ax.set_xticks(trend_sens)
    ax.set_xlim(min(trend_sens)-1, max(trend_sens)+1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle="--", lw=0.5, alpha=0.7)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)

    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5,1.12))
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

    # 1 ─ crisis windows on PuE and SEI
    _, dd_pu  = calculate_cumulative_returns_and_drawdowns(pu_ret)
    _, dd_sei = calculate_cumulative_returns_and_drawdowns(sei_ret)
    pu_cr  = detect_crisis_periods(dd_pu,  threshold)
    sei_cr = detect_crisis_periods(dd_sei, threshold)

    # 2 ─ cumulative TSM indices on quarter‑ends
    tags = {"V Fast", "Fast", "Med", "Slow", "V Slow"}
    cum_q = {}
    for col in hedge_monthly:
        if ("momentum" in col.lower()) or (col.strip() in tags):
            cum = (1 + hedge_monthly[col].dropna()).cumprod()
            cum_q[col] = cum.asfreq("QE-DEC", method="ffill").reindex(idx_q)

    if not cum_q:
        st.warning("No TSM data found.");  return

    # 3 ─ keep crises fully covered by all overlays
    def covered(cr):
        s, t = cr["Start"], cr["Trough"]
        return all((s in srs.index and t in srs.index
                    and not (pd.isna(srs.loc[s]) or pd.isna(srs.loc[t])))
                   for srs in cum_q.values())

    pu_cr  = [c for c in pu_cr  if covered(c)]
    sei_cr = [c for c in sei_cr if covered(c)]

    pt = lambda ser, cr: ser.loc[cr["Trough"]] / ser.loc[cr["Start"]] - 1

    rows = []
    for name in cum_q:
        cum_qtr = cum_q[name].dropna()
        if cum_qtr.empty:
            continue

        # crisis pay‑offs
        pu_vals  = [pt(cum_qtr, c) for c in pu_cr]
        sei_vals = [pt(cum_qtr, c) for c in sei_cr]

        med_pu,  med_sei  = map(np.nanmedian, (pu_vals, sei_vals))
        mean_pu, mean_sei = map(np.nanmean,  (pu_vals, sei_vals))

        # ── ORIGINAL carry: CAGR of MONTHLY returns from hedge_monthly
        ret_m  = hedge_monthly[name].dropna()
        years  = len(ret_m) / 12
        carry  = (1 + ret_m).prod()**(1 / years) - 1
        # -------------------------------------------------------------------

        pcr_pu  = protection_cost_ratio(med_pu,  carry)
        pcr_sei = protection_cost_ratio(med_sei, carry)

        rows.append({
            "Strategy": name,
            "Speed": ("Fast" if "fast" in name.lower()
                      else "Slow" if any(x in name.lower() for x in ("slow", "med"))
                      else "Other"),
            "Median PuE":  med_pu,
            "Median SEI":  med_sei,
            "Mean PuE":    mean_pu,
            "Mean SEI":    mean_sei,
            "Rel Median":  None if med_pu  == 0 else med_sei  / med_pu  - 1,
            "Rel Mean":    None if mean_pu == 0 else mean_sei / mean_pu - 1,
            "PCR PuE":     pcr_pu,
            "PCR SEI":     pcr_sei,
            "Rel PCR":     None if pcr_pu  == 0 else pcr_sei  / pcr_pu  - 1,
            "Annualised Carry %": carry * 100,
        })

    if not rows:
        st.info("No crises with full TSM coverage."); return

    df = pd.DataFrame(rows).set_index("Strategy")

    pct1 = lambda x: "NA" if pd.isna(x) else f"{x*100:.1f}%"
    two  = "{:.2f}".format
    fmt  = {"Median PuE": pct1, "Median SEI": pct1,
            "Mean PuE":   pct1, "Mean SEI":   pct1,
            "Rel Median": pct1, "Rel Mean":   pct1,
            "PCR PuE":    two,  "PCR SEI":    two,  "Rel PCR": two,
            "Annualised Carry %": "{:.2f}"}

    st.subheader("Fast vs Slow TSM (10 % drawdown, relative improvements)")
    st.dataframe(df.style.format(fmt), height=420)



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

    for thr in [0.10, 0.075, 0.125]:
        st.header(f"{int(thr*100)}% Threshold")
        process_threshold(thr, cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens)

    process_top_crises(cum_pe, dd_pe, cum_sei, dd_sei, cum_hedge, trend_sens, n=4)

    # ------------------ TSM speed table ------------------------------
    tsm_speed_table(idx_q, pe_ret, sei_ret, hedge_df, 0.10)


if __name__ == "__main__":
    main()
