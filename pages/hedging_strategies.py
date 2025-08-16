# hedging_strategies.py

import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

# We'll import your main functions from the existing app (assuming the same folder structure)
# Make sure these functions are present in your local 'streamlit_app.py'
from streamlit_app import (
    load_and_preprocess_data,   # Loads allocations + quarterly asset returns
    load_individual_endowments, # Loads individual_endowments.csv
    unify_timeframe,
    map_allocations_to_periods,
    calculate_cumulative_and_dd
)

###############################################################################
#                  LOAD HEDGING STRATEGIES CSV
###############################################################################
def load_hedging_strategies():
    """
    Load CSV "data/hedging_strategies.csv" with columns like:
    Date;Global Macro;Hedge Funds;Tail Risk Hedge Funds;V Fast;Fast;Med;Slow;V Slow
    Convert to decimals, set Date as index (month-end).
    After loading, rename:
      'V Fast' -> 'Time Series Momentum (Very Fast)'
      'Fast'   -> 'Time Series Momentum (Fast)'
      'Med'    -> 'Time Series Momentum (Med)'
      'Slow'   -> 'Time Series Momentum (Slow)'
      'V Slow' -> 'Time Series Momentum (Very Slow)'
    Then add TSM Basket as the equal-weighted average of the five TSM sleeves.
    """
    file_path = "data/hedging_strategies.csv"
    df = pd.read_csv(file_path, sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)

    # Move index to end of the month
    df.index = df.index + pd.offsets.MonthEnd(0)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) and x != '' else np.nan
        )

    df.sort_index(inplace=True)

    # Rename columns for TSM speeds
    rename_map = {
        'V Fast': 'Time Series Momentum (Very Fast)',
        'Fast': 'Time Series Momentum (Fast)',
        'Med': 'Time Series Momentum (Med)',
        'Slow': 'Time Series Momentum (Slow)',
        'V Slow': 'Time Series Momentum (Very Slow)'
    }
    df.rename(columns=rename_map, inplace=True)

    # Add TSM Basket (equal-weighted average)
    tsm_cols = list(rename_map.values())
    df['Time Series Momentum (Basket)'] = df[tsm_cols].mean(axis=1)

    return df

###############################################################################
#               CRISIS DETECTION (SYNTHETIC ENDOWMENT)
###############################################################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.10):
    """
    Identify crisis periods in the Synthetic Endowment Index's drawdown if ≥ threshold (10 %).
    For each crisis, the 'Start' is the last date before the drawdown begins (when the drawdown
    was zero or positive), the 'Trough' is when the maximum drawdown is reached, and the 'End'
    date is when the drawdown fully recovers back to 0 %. Returns a list of dicts:
       { 'Start', 'Trough', 'End', 'Max Drawdown' }.
    Crises where the peak and trough dates are the same are skipped.
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
            # Set the peak date as the previous date if available and non‑negative;
            # otherwise, use the current date.
            if idx > 0 and ds.iloc[idx - 1] >= 0:
                start_date = ds.index[idx - 1]
            else:
                start_date = date
            trough_date = date
            max_dd = dd_val
        elif in_crisis:
            if dd_val < max_dd:
                max_dd = dd_val
                trough_date = date
            if dd_val == 0:
                in_crisis = False
                end_date = date
                # Only add the crisis if the peak and trough are distinct and dd ≤ ‑threshold
                if max_dd <= -threshold and start_date != trough_date:
                    crises.append({
                        'Start': start_date,
                        'Trough': trough_date,
                        'End': end_date,
                        'Max Drawdown': max_dd
                    })
                start_date, trough_date, max_dd = None, None, 0.0

    if in_crisis:
        end_date = ds.index[-1]
        if max_dd <= -threshold and start_date != trough_date:
            crises.append({
                'Start': start_date,
                'Trough': trough_date,
                'End': end_date,
                'Max Drawdown': max_dd
            })
    return crises

def create_crisis_return_panel(cum_dict, crises, hedge_keys):
    """
    Creates a third panel showing cumulative performance of hedging strategies
    during each crisis period, starting at 0% for each crisis.
    """
    fig, ax = plt.subplots(figsize=(8, 2.8))

    for key in hedge_keys:
        series = cum_dict.get(key, None)
        if series is None or series.empty:
            continue

        for crisis in crises:
            s_dt, e_dt = crisis["Start"], crisis["End"]
            if s_dt not in series.index or e_dt not in series.index:
                continue

            sub = series.loc[s_dt:e_dt].dropna()
            if sub.empty:
                continue

            base = sub.iloc[0]
            rel = sub / base - 1.0
            ax.plot(rel.index, rel.values, label=key)

    ax.set_ylabel("Crisis Return", fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelsize=8)
    ax.set_ylim(auto=True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title("Hedging Strategy Performance During Crisis Periods", fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    if len(hedge_keys) <= 7:
        ax.legend(loc='best', fontsize=7)

    fig.tight_layout()
    return fig

###############################################################################
#  (NEW)  BUILD PER-CRISIS “RESET” SERIES  FOR THE 3-PANEL CHART
###############################################################################
def _build_crisis_return_series(cum_series, crises):
    """
    For one strategy’s cumulative-NAV series, return a copy whose value
    *resets to 0* at each crisis Start and then tracks the cumulative %
    return until the crisis End.  Outside crisis windows the series is NaN,
    so the bottom panel appears blank (= white).
    """
    if cum_series.empty or not crises:
        return pd.Series(dtype=float)

    # Start with an all-NaN series, same index as cum_series
    out = pd.Series(np.nan, index=cum_series.index)

    for c in crises:
        s, e = c["Start"], c["End"]
        if s not in cum_series.index or e not in cum_series.index:
            continue  # skip if dates missing

        # Sub-series for the crisis window
        sub = cum_series.loc[s:e]
        base = sub.iloc[0]
        if np.isclose(base, 0):
            continue
        # Normalise: (cum / base) – 1  → starts at 0 %
        out.loc[s:e] = sub / base - 1.0

    return out


def build_crisis_return_dict(cum_dict, crises, hedge_keys):
    """
    Create {strategy: crisis-return-series}.  Only hedging strategies
    (hedge_keys) are included; SEI / PuE are *not* part of this dict.
    """
    crisis_dict = {}
    for k in hedge_keys:
        crisis_dict[k] = _build_crisis_return_series(cum_dict[k], crises)
    return crisis_dict


###############################################################################
#     PIVOT TABLE (REVISED): SHOW HEDGING STRATEGIES' PEAK‑>TROUGH PERF
###############################################################################
def pivot_crises_quarterly(rep_crises, cum_dict):
    """
    Build a table for each crisis (≥ 10 % on Synthetic Endowment Index).

    Revised to:
      • Report hedging performance from Start (peak) -> Trough only
      • Include a new "Recovery Date" row
      • Keep time‑to‑recovery logic for the Synthetic Endowment Index
        (peak -> trough -> full recovery).
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    # We add a "Recovery Date" row after Trough Date:
    row_labels = [
        "Beginning Date",   # crisis start
        "Trough Date",
        "Recovery Date",    # new row
        "Synthetic Endowment Index Max Drawdown",
        "Drawdown Length (quarters)",
        "Time to Recovery (quarters)",
        "midrule1",
        "Public Equity Drawdown (Peak->Trough)",
        "Public Equity Time to Recovery",
        "midrule2"
    ]

    # Identify hedging columns except "Synthetic Endowment Index" and "Public Equity"
    exclude_labels = ["Synthetic Endowment Index", "Public Equity"]
    hedge_strategies = [
        k for k in cum_dict.keys() if k not in exclude_labels and not cum_dict[k].empty
    ]

    final_rows = row_labels + hedge_strategies
    data = {c: [""] * len(final_rows) for c in col_names}

    def approximate_quarters_diff(ts_start, ts_end):
        days_diff = (ts_end - ts_start).days
        if days_diff < 0:
            return 0
        months = days_diff / 30.4375
        q_float = months / 3.0
        q_rounded = int(round(q_float))
        return max(1, q_rounded)

    def find_time_to_recovery_q(cum_series, start_d, trough_d):
        """
        If we want to see how many quarters from Trough until
        the series gets back above the Start's value, do that here.
        """
        if start_d not in cum_series.index or trough_d not in cum_series.index:
            return None
        peak_val = cum_series.loc[start_d]
        subset = cum_series.loc[trough_d:]
        rec_idx = subset[subset >= peak_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_quarters_diff(trough_d, rec_idx[0])

    syn_cum = cum_dict.get("Synthetic Endowment Index", pd.Series(dtype=float))
    pe_cum  = cum_dict.get("Public Equity", pd.Series(dtype=float))

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        s_dt = crisis['Start']
        t_dt = crisis['Trough']
        e_dt = crisis['End']
        mdd = crisis['Max Drawdown']

        # Rows 0..2 => Basic crisis dates
        data[cname][0] = str(s_dt.date())  # Beginning Date
        data[cname][1] = str(t_dt.date())  # Trough Date
        data[cname][2] = str(e_dt.date())  # Recovery Date (new)

        # Row 3..5 => Synthetic Endowment drawdown stats
        data[cname][3] = f"{mdd:.1%}"  # Max Drawdown
        len_q = approximate_quarters_diff(s_dt, t_dt)
        data[cname][4] = f"{len_q}"
        if syn_cum is not None and not syn_cum.empty \
           and (s_dt in syn_cum.index) and (t_dt in syn_cum.index):
            r_ttr = find_time_to_recovery_q(syn_cum, s_dt, t_dt)
            data[cname][5] = f"{r_ttr}" if r_ttr else "n/a"
        else:
            data[cname][5] = "n/a"

        # row 6 => midrule1
        data[cname][6] = "midrule1"

        # row 7..8 => Public Equity stats, peak->trough
        if pe_cum is not None and not pe_cum.empty \
           and (s_dt in pe_cum.index) and (t_dt in pe_cum.index):
            pe_pk = pe_cum.loc[s_dt]
            pe_th = pe_cum.loc[t_dt]
            eq_dd = (pe_th - pe_pk) / pe_pk
            data[cname][7] = f"{eq_dd:.1%}" if eq_dd < -1e-9 else "n/a"

            eq_ttr = find_time_to_recovery_q(pe_cum, s_dt, t_dt)
            data[cname][8] = f"{eq_ttr}" if eq_ttr else "n/a"
        else:
            data[cname][7] = "n/a"
            data[cname][8] = "n/a"

        # row 9 => midrule2
        data[cname][9] = "midrule2"

        # Hedge strategies => measure from Start->Trough
        offset = len(row_labels)
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset + idx_h
            strat_cum = cum_dict[strat_name]
            if s_dt in strat_cum.index and t_dt in strat_cum.index:
                start_val = strat_cum.loc[s_dt]
                trough_val = strat_cum.loc[t_dt]
                if start_val <= 1e-9:
                    data[cname][row_idx] = "n/a"
                else:
                    perf = (trough_val / start_val) - 1.0
                    data[cname][row_idx] = f"{perf:.1%}"
            else:
                data[cname][row_idx] = "n/a"

    return pd.DataFrame(data, index=final_rows)

def export_table_to_latex_pivot(df, description="Quarterly Crisis Table + Hedge Perf"):
    """
    Convert pivot table to LaTeX code.
    Introduce midrule after row labeled 'midrule1' and 'midrule2'.
    """
    columns = df.columns.tolist()
    row_labels = df.index.tolist()

    align_str = "l" + "".join(["c"] * len(columns))

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\"  
        r"\footnotesize{Crises identified at a 10\% threshold on the Synthetic Endowment Index. "
        r"Public Equity lines are divided by midrules. Hedging strategies show peak‑to‑trough "
        r"performance.}}"
    )
    lines.append(r"\label{table:hedge_crisis}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for row_lbl in row_labels:
        if row_lbl == "midrule1" or row_lbl == "midrule2":
            lines.append(r"\midrule")
            continue

        row_vals = [row_lbl]
        for col in columns:
            val = df.loc[row_lbl, col]
            if isinstance(val, str):
                val = val.replace("%", r"\%")  # latex escape
            row_vals.append(val)
        line = " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

###############################################################################
#    NEW: PIVOT TABLE WITH 90/10 OVERLAY ROWS, PEAK‑>TROUGH
###############################################################################
def pivot_crises_quarterly_expanded(rep_crises, cum_dict):
    """
    Build a pivot table for quarterly crises structured in three blocks:
      Block 1: Synthetic Endowment Index drawdowns (Peak->Trough->Recovery row).
      Block 2: Hedge Strategy (peak‑to‑trough).
      Block 3: Synthetic Endowment + 10 % Overlay (peak‑to‑trough).

    The difference from the original version:
      • We measure strategy performance Start->Trough (not Start->End).
      • We add a "Recovery Date" row but do not use it for the returns.
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    # Block 1 row labels (add "Recovery Date" after Trough Date).
    base_rows = [
        "Peak Date",
        "Trough Date",
        "Recovery Date",
        "Max Drawdown",
        "Drawdown Length (Qtrs)",
        "Time to Recovery (Qtrs)",
        "midrule1"  # marker
    ]

    # Identify hedge columns
    exclude_labels = ["Synthetic Endowment Index", "Public Equity"]
    hedge_strategies = [
        k for k in cum_dict.keys() if k not in exclude_labels and not cum_dict[k].empty
    ]

    # Block 2: Hedge Strategy performance
    block2_rows = hedge_strategies
    # Block 3: 90/10 overlay
    block3_rows = [f"Synthetic Endowment + 10% {strat}" for strat in hedge_strategies]

    final_rows = base_rows + block2_rows + ["midrule2"] + block3_rows
    data = {c: [""] * len(final_rows) for c in col_names}

    def approximate_quarters_diff(ts_start, ts_end):
        days_diff = (ts_end - ts_start).days
        if days_diff < 0:
            return 0
        months = days_diff / 30.4375
        q_float = months / 3.0
        q_rounded = int(round(q_float))
        return max(1, q_rounded)

    def find_time_to_recovery_q(cum_series, start_d, trough_d):
        if start_d not in cum_series.index or trough_d not in cum_series.index:
            return None
        peak_val = cum_series.loc[start_d]
        subset = cum_series.loc[trough_d:]
        rec_idx = subset[subset >= peak_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_quarters_diff(trough_d, rec_idx[0])

    def compute_overlay_performance(syn_series, hedge_series, s_dt, t_dt, weight=0.1):
        """
        Compute 90/10 overlay from s_dt to t_dt. If there's only one or two data points
        per subperiod, this is effectively just a direct ratio or a small chain of returns.
        """
        if (s_dt not in syn_series.index) or (t_dt not in syn_series.index) \
           or (s_dt not in hedge_series.index) or (t_dt not in hedge_series.index):
            return None

        syn_sub = syn_series.loc[s_dt:t_dt]
        hedge_sub = hedge_series.loc[s_dt:t_dt]

        # If subperiod has only 1 or 2 points, direct ratio approach:
        if len(syn_sub) < 2 or len(hedge_sub) < 2:
            syn_return = syn_sub.iloc[-1]/syn_sub.iloc[0] - 1.0
            hedge_return = hedge_sub.iloc[-1]/hedge_sub.iloc[0] - 1.0
            return (1-weight)*syn_return + weight*hedge_return

        # Otherwise, step through each quarter
        syn_ret = syn_sub.pct_change().dropna()
        hedge_ret = hedge_sub.pct_change().dropna()
        df = pd.DataFrame({"syn": syn_ret, "hedge": hedge_ret}).dropna()
        if df.empty:
            return None

        combined_ret = (1-weight)*df["syn"] + weight*df["hedge"]
        cum_ = (1 + combined_ret).cumprod()
        return cum_.iloc[-1] - 1.0

    syn_cum = cum_dict.get("Synthetic Endowment Index", pd.Series(dtype=float))

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        s_dt = crisis['Start']
        t_dt = crisis['Trough']
        e_dt = crisis['End']
        mdd = crisis['Max Drawdown']

        # Block 1: Synthetic Endowment
        data[cname][0] = str(s_dt.date())  # Peak
        data[cname][1] = str(t_dt.date())  # Trough
        data[cname][2] = str(e_dt.date())  # Recovery
        data[cname][3] = f"{mdd:.1%}"
        len_q = approximate_quarters_diff(s_dt, t_dt)
        data[cname][4] = f"{len_q}"
        if (s_dt in syn_cum.index) and (t_dt in syn_cum.index):
            syn_ttr = find_time_to_recovery_q(syn_cum, s_dt, t_dt)
            data[cname][5] = f"{syn_ttr}" if syn_ttr else "n/a"
        else:
            data[cname][5] = "n/a"

        # Block 2: Hedge Strategy (peak->trough)
        offset_block2 = len(base_rows)
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset_block2 + idx_h
            strat_cum = cum_dict[strat_name]
            if s_dt in strat_cum.index and t_dt in strat_cum.index:
                start_val = strat_cum.loc[s_dt]
                trough_val = strat_cum.loc[t_dt]
                if start_val <= 1e-9:
                    data[cname][row_idx] = "n/a"
                else:
                    perf = (trough_val/start_val) - 1.0
                    data[cname][row_idx] = f"{perf:.1%}"
            else:
                data[cname][row_idx] = "n/a"

        # Block 3: 90/10 overlays (peak->trough)
        offset_block3 = len(base_rows) + len(hedge_strategies) + 1
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset_block3 + idx_h
            overlay_perf = compute_overlay_performance(syn_cum, cum_dict[strat_name], s_dt, t_dt, weight=0.1)
            if overlay_perf is None:
                data[cname][row_idx] = "n/a"
            else:
                data[cname][row_idx] = f"{overlay_perf:.1%}"

    return pd.DataFrame(data, index=final_rows)

def export_table_to_latex_pivot_expanded(
    df,
    description="Quarterly Crisis Periods + Hedge Strategy Cumulative Performance (90/10 Overlays)"
):
    """
    Convert the pivot table to LaTeX code formatted for the Journal of Finance style,
    showing the new row "Recovery Date," and blocks for Hedge Strategy vs. Overlays.
    """
    columns = df.columns.tolist()
    align_str = "l" + "".join(["c"] * len(columns))

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\"  
        r"\footnotesize{Major crisis periods (≥ 10 % drawdown on the Synthetic Endowment Index). "
        r"Hedge strategy returns and 90/10 overlays are measured peak‑to‑trough.}}"
    )
    lines.append(r"\label{table:hedge_crisis_jf}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header for crisis columns
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # We'll place some textual blocks. We'll assume the "midrule1" or "midrule2" markers
    # exist in the index or we do it by row offset.
    row_labels = df.index.tolist()

    # Insert a "Synthetic Endowment Index" label at the top block
    lines.append(r"\textbf{Synthetic Endowment Index} \\")
    for row_lbl in row_labels:
        if row_lbl == "midrule1":
            lines.append(r"\midrule")
            lines.append(r"\textbf{Hedge Strategy (Peak‑to‑Trough Returns)} \\")
            continue
        if row_lbl == "midrule2":
            lines.append(r"\midrule")
            lines.append(r"\textbf{Synthetic Endowment + 10\% Overlay Performance} \\")
            continue

        # Indent the row label a bit
        display_lbl = r"\quad " + row_lbl
        row_vals = [display_lbl]
        for col in columns:
            val = df.loc[row_lbl, col]
            if isinstance(val, str):
                val = val.replace("%", r"\%")
            row_vals.append(val)
        lines.append(" & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

###############################################################################
#                HELPERS FOR NEW SPLIT CHARTS
###############################################################################
def _linestyle_for(label: str):
    """Dashed for Synthetic Endowment & Public Equity, solid otherwise."""
    if label == "Synthetic Endowment Index":
        return dict(linestyle="--", linewidth=1.2, color="black")
    if label == "Public Equity":
        return dict(linestyle="--", linewidth=1.2, color="dimgray")
    return dict(linestyle="-",  linewidth=1.2)  # hedge strategies

def create_split_chart(cum_dict, dd_dict, hedge_keys, crises=None, title=""):
    """
    Two-row Matplotlib figure (cumulative & drawdown) for a subset of hedging
    strategies (hedge_keys). Synthetic Endowment Index and Public Equity are
    always included and plotted dashed; hedge strategies are plotted solid.
    """
    ref_keys = [k for k in ["Synthetic Endowment Index", "Public Equity"] if k in cum_dict]
    all_keys = ref_keys + hedge_keys

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

    # Top panel – cumulative
    for k in all_keys:
        ser = cum_dict[k].dropna()
        if ser.empty:
            continue
        ax1.plot(ser.index, ser.values, label=k, **_linestyle_for(k))
    ax1.set_ylabel("Cumulative Return", fontsize=8)
    ax1.tick_params(labelsize=8)

    # Bottom panel – drawdown
    for k in all_keys:
        ser = dd_dict[k].dropna()
        if ser.empty:
            continue
        ax2.plot(ser.index, ser.values, label=k, **_linestyle_for(k))
    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.tick_params(labelsize=8)

    # Crisis shading
    if crises:
        for crisis in crises:
            ax1.axvspan(crisis['Start'], crisis['End'], color='lightgray', alpha=0.3, zorder=0)
            ax2.axvspan(crisis['Start'], crisis['End'], color='lightgray', alpha=0.3, zorder=0)

    # X-axis formatting
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Legend without duplicates
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
    for h, l in zip(*ax2.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=2, frameon=False, fontsize=8)

    fig.tight_layout(pad=1.5)
    plt.subplots_adjust(bottom=0.20)
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    return fig

###############################################################################
#                          3-PANEL CHART CREATOR  (revised)
###############################################################################
def create_three_panel_chart(
    cum_dict,
    dd_dict,
    crisis_ret_dict,
    crises=None,
    crisis_ylim=(-0.10, 0.40)      # fixed y-scale for bottom panel
):
    """
    Three stacked panels:
      (1) cumulative returns
      (2) drawdowns (SEI & Public Equity only)
      (3) cumulative crisis-period returns
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import matplotlib.dates as mdates

    # ----- helper for consistent line style -----
    def _ls(label):
        if label == "Synthetic Endowment Index":
            return dict(linestyle="--", linewidth=1.2, color="black")
        if label == "Public Equity":
            return dict(linestyle="--", linewidth=1.2, color="dimgray")
        return dict(linestyle="-",  linewidth=1.2)

    # slightly shorter figure + reduced h-space
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8, 5),
        gridspec_kw={'hspace': 0.06}
    )

    # ---------------- (1) cumulative ----------------
    for lbl, ser in cum_dict.items():
        if ser.empty: continue
        ax1.plot(ser.index, ser.values, label=lbl, **_ls(lbl))
    ax1.set_ylabel("Cumulative\nReturn", fontsize=8)
    ax1.tick_params(labelsize=8)

    # ---------------- (2) drawdown ------------------
    for lbl in ("Synthetic Endowment Index", "Public Equity"):
        if lbl in dd_dict and not dd_dict[lbl].empty:
            ax2.plot(dd_dict[lbl].index, dd_dict[lbl].values, **_ls(lbl))
    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.tick_params(labelsize=8)

    # ---------------- (3) crisis returns ------------
    for lbl, ser in crisis_ret_dict.items():
        if ser.empty: continue
        ax3.plot(ser.index, ser.values, label=lbl, **_ls(lbl))
    ax3.set_ylabel("Crisis\nReturn", fontsize=8)
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.set_ylim(*crisis_ylim)
    ax3.tick_params(labelsize=8)

    # --------- crisis shading for all panels --------
    if crises:
        for c in crises:
            for ax in (ax1, ax2, ax3):
                ax.axvspan(c['Start'], c['End'],
                           color='lightgray', alpha=0.3, zorder=0)

    # --------------- x-axis formatting --------------
    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ---------------- overall legend ----------------
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.06), ncol=3,
        frameon=False, fontsize=8
    )

    # tight layout with smaller padding
    fig.tight_layout(pad=0.6)
    plt.subplots_adjust(bottom=0.18)   # leaves room for legend

    return fig


###############################################################################
#                              MAIN APP
###############################################################################
def main():
    st.title("Quarterly Crisis Period Table + Cumulative Hedge Performance (Peak->Trough)")

    # 1) Load main data
    allocations, ret_q_full = load_and_preprocess_data()
    # Use a copy of ret_q_full to create the unified timeframe for allocations‑based returns
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q_full.copy())
    idx_q = pd.date_range(q_start, q_end, freq='Q-DEC')

    # Build Synthetic Endowment Index (Q) using the unified timeframe
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)
    valid_cols = hist_alloc_q.columns.intersection(ret_q.columns)
    hist_alloc_q, ret_q_hist = hist_alloc_q[valid_cols], ret_q[valid_cols]
    syn_index_q = (hist_alloc_q * ret_q_hist).sum(axis=1).dropna()
    syn_cum, syn_dd = calculate_cumulative_and_dd(syn_index_q)

    # 2) Load hedging strategies (monthly) => unify => resample to Q => build cumulative
    hedge_df = load_hedging_strategies()
    hedge_df = hedge_df[(hedge_df.index >= q_start) & (hedge_df.index <= q_end)]
    # resample to quarterly, picking last monthly data in each quarter
    hedge_q = hedge_df.resample('Q').last().dropna(how='all')

    # Build cum_dict for pivot (strictly quarterly data)
    cum_dict = {"Synthetic Endowment Index": syn_cum}

    # For Public Equity, ensure we also convert it to quarterly if present
    if "Public Equity" in ret_q_full.columns:
        pe_ = ret_q_full["Public Equity"].dropna()
        pe_q = pe_.resample('Q').last().dropna()
        pe_cum, _ = calculate_cumulative_and_dd(pe_q)
        cum_dict["Public Equity"] = pe_cum

    for col in hedge_q.columns:
        col_series = hedge_q[col].dropna()
        if not col_series.empty:
            c_, _ = calculate_cumulative_and_dd(col_series)
            cum_dict[col] = c_

    # 3) Identify crises using Synthetic Endowment Index drawdowns (≥10 %)
    crises = find_crisis_periods_for_rep_endowment(syn_dd, threshold=0.10)

    # -------------------------------------------------------------------------
    # 4) Revised pivot table (peak->trough) + "Recovery Date"
    pivoted = pivot_crises_quarterly(crises, cum_dict)
    st.subheader("Quarterly Crisis Table with Hedging Strategies (Peak->Trough, ≥ 10 % Drawdown)")
    if pivoted.empty:
        st.warning("No crises found or no data to display.")
    else:
        st.dataframe(pivoted)
        latex_code = export_table_to_latex_pivot(
            pivoted,
            description="Quarterly Crisis Periods + Hedge Strategy Peak->Trough Performance"
        )
        st.write("**LaTeX (Revised Original Pivot):**")
        st.code(latex_code, language="latex")

    # -------------------------------------------------------------------------
    # 5) Expanded pivot table with 90/10 rows, also peak->trough
    expanded_pivot = pivot_crises_quarterly_expanded(crises, cum_dict)
    st.subheader("Quarterly Crisis Table with Hedging Strategies + 90/10 Overlays (Peak->Trough, ≥ 10 %)")
    if expanded_pivot.empty:
        st.warning("No crises found or no data to display (expanded).")
    else:
        st.dataframe(expanded_pivot)
        latex_expanded = export_table_to_latex_pivot_expanded(
            expanded_pivot,
            description="Quarterly Crisis Periods + Hedge Strategy (Peak->Trough) + 90/10 Overlays"
        )
        st.write("**LaTeX (Expanded Pivot with Recovery Date):**")
        st.code(latex_expanded, language="latex")

    # -------------------------------------------------------------------------
    # 6) ORIGINAL JF‑style Chart with shading
    dd_dict = {}
    for lbl, series_c in cum_dict.items():
        if series_c.empty:
            dd_dict[lbl] = pd.Series(dtype=float)
        else:
            run_max = series_c.cummax()
            dd_ = (series_c - run_max) / run_max
            dd_dict[lbl] = dd_

    st.subheader("Cumulative & Drawdown Chart (JF‑Style) with Crisis Shading")
    fig = create_split_chart(
        cum_dict,
        dd_dict,
        hedge_keys=[k for k in cum_dict if k not in ("Synthetic Endowment Index", "Public Equity")],
        crises=crises,
        title="All Hedging Strategies"
    )
    st.pyplot(fig)

    # -------------------------------------------------------------------------
    # 7) NEW split charts (TSM versus other hedges)
    tsm_keys = [k for k in cum_dict if "Time Series Momentum" in k]
    other_keys = [k for k in cum_dict
                  if k not in tsm_keys and k not in ("Synthetic Endowment Index", "Public Equity")]

    st.subheader("Split Chart – Time Series Momentum Strategies")
    if tsm_keys:
        fig_tsm = create_split_chart(cum_dict, dd_dict, hedge_keys=tsm_keys,
                                     crises=crises)
        st.pyplot(fig_tsm)
    else:
        st.info("No Time Series Momentum strategies available.")

    st.subheader("Split Chart – Other Hedging Strategies")
    if other_keys:
        fig_other = create_split_chart(cum_dict, dd_dict, hedge_keys=other_keys,
                                       crises=crises)
        st.pyplot(fig_other)
    else:
        st.info("No other hedging strategies available.")

    # -------------------------------------------------------------------------
    # 8) Informational footer
    min_date, max_date = None, None
    for series_data in cum_dict.values():
        if not series_data.empty:
            smin = series_data.index.min()
            smax = series_data.index.max()
            min_date = smin if min_date is None else min(min_date, smin)
            max_date = smax if max_date is None else max(max_date, smax)

    if min_date and max_date:
        date_range_info = (
            f"The series are displayed from {min_date.strftime('%Y-%m-%d')} "
            f"to {max_date.strftime('%Y-%m-%d')}, at a quarterly frequency."
        )
    else:
        date_range_info = "No valid date range found (empty data)."

    st.write(
        "All hedge performances are measured from the crisis peak date to the trough date. "
        "A new 'Recovery Date' row is included for reference, but does not affect the "
        "performance calculations. The table's 'Time to Recovery' columns refer to how many "
        "quarters it took for the Synthetic Endowment (or Public Equity) to regain its peak "
        f"levels after the trough.\n\n**Date Range**: {date_range_info}"
    )

    ###############################################################################
    # 9)  THREE-PANEL CHARTS:  TSM   vs.   Other Hedging Strategies
    ###############################################################################

    tsm_keys   = [k for k in cum_dict if "Time Series Momentum" in k]
    other_keys = [k for k in cum_dict
                if k not in tsm_keys and k not in ("Synthetic Endowment Index",
                                                    "Public Equity")]

    # ========== (A) TSM-only chart ==============================================
    # --- keep only SEI, PuE, and TSM sleeves ------------------------------------
    cum_tsm = {k: v for k, v in cum_dict.items()
            if k in ("Synthetic Endowment Index", "Public Equity") or k in tsm_keys}
    dd_tsm  = {k: v for k, v in dd_dict.items() if k in cum_tsm}

    crisis_ret_tsm = build_crisis_return_dict(cum_tsm, crises, hedge_keys=tsm_keys)

    st.subheader("3-Panel Chart – Time-Series Momentum Strategies")
    if tsm_keys:
        fig_3p_tsm = create_three_panel_chart(
            cum_tsm,
            dd_tsm,
            crisis_ret_tsm,
            crises=crises
        )
        st.pyplot(fig_3p_tsm)
    else:
        st.info("No Time-Series Momentum strategies available.")

    # ========== (B) Other-hedges chart ==========================================
    # --- keep only SEI, PuE, and non-TSM hedges ---------------------------------
    cum_other = {k: v for k, v in cum_dict.items()
                if k in ("Synthetic Endowment Index", "Public Equity") or k in other_keys}
    dd_other  = {k: v for k, v in dd_dict.items() if k in cum_other}

    crisis_ret_other = build_crisis_return_dict(cum_other, crises,
                                                hedge_keys=other_keys)

    st.subheader("3-Panel Chart – Other Hedging Strategies")
    if other_keys:
        fig_3p_other = create_three_panel_chart(
            cum_other,
            dd_other,
            crisis_ret_other,
            crises=crises
        )
        st.pyplot(fig_3p_other)
    else:
        st.info("No other hedging strategies available.")



if __name__ == "__main__":
    main()
