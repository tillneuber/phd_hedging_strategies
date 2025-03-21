# Filename: hedging_strategies.py

import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

# We'll import your main functions from the existing app (assuming the same folder structure)
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
    so that all five speeds are recognized as TSM strategies.
    """
    file_path = "data/hedging_strategies.csv"
    df = pd.read_csv(file_path, sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    # Move index to end of month
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

    return df


###############################################################################
#               CRISIS DETECTION (REP ENDOWMENT)
###############################################################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.05):
    """
    Identify crisis periods in the Representative Endowment's drawdown if ≥ threshold (5%).
    For each crisis, the "Peak Date" is defined as the last date before the drawdown begins (when the drawdown was zero or positive),
    and the "Trough Date" is when the maximum drawdown is reached.
    Crises where the peak and trough dates are the same are skipped.
    Returns a list of dicts: { 'Start', 'Trough', 'End', 'Max Drawdown' }.
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
            # Set the peak date as the previous date if available and non-negative;
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
                # Only add the crisis if the peak and trough are distinct.
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



###############################################################################
#     PIVOT TABLE (ORIGINAL): SHOW HEDGING STRATEGIES' CUMULATIVE PERF
###############################################################################
def pivot_crises_quarterly(rep_crises, cum_dict):
    """
    Build a table for each crisis (≥5% on Representative Endowment).
    Columns: Crisis1, Crisis2, ...
    Rows:
      1) Beginning Date
      2) Trough Date
      3) Rep. Endowment Max Drawdown
      4) Drawdown Length (quarters)
      5) Time to Recovery (quarters)
      6) midrule1
      7) Public Equity Drawdown
      8) Public Equity Time to Recovery
      9) midrule2
      Then each hedging strategy => single row with the cumulative performance
      from crisis Start to crisis End.
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    row_labels = [
        "Beginning Date",
        "Trough Date",
        "Rep. Endowment Max Drawdown",
        "Drawdown Length (quarters)",
        "Time to Recovery (quarters)",
        "midrule1",
        "Public Equity Drawdown",
        "Public Equity Time to Recovery",
        "midrule2"
    ]

    # Identify hedging columns except "Representative Endowment" and "Public Equity"
    exclude_labels = ["Representative Endowment", "Public Equity"]
    hedge_strategies = [
        k for k in cum_dict.keys() if k not in exclude_labels and not cum_dict[k].empty
    ]

    row_entries = list(hedge_strategies)  # Each hedge in a separate row
    final_rows = row_labels + row_entries

    data = {c: [""] * len(final_rows) for c in col_names}

    def approximate_quarters_diff(ts_start, ts_end):
        days_diff = (ts_end - ts_start).days
        if days_diff < 0:
            return 0
        months = days_diff / 30.4375
        return max(1, int(math.ceil(months / 3)))

    def find_time_to_recovery_q(cum_series, start_d, trough_d):
        if start_d not in cum_series.index or trough_d not in cum_series.index:
            return None
        peak_val = cum_series.loc[start_d]
        subset = cum_series.loc[trough_d:]
        rec_idx = subset[subset >= peak_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_quarters_diff(trough_d, rec_idx[0])

    rep_cum = cum_dict.get("Representative Endowment", pd.Series(dtype=float))
    pe_cum  = cum_dict.get("Public Equity", pd.Series(dtype=float))

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        s_dt = crisis['Start']
        t_dt = crisis['Trough']
        e_dt = crisis['End']
        mdd = crisis['Max Drawdown']

        # Rows 0,1,2,3,4 = Basic crisis info
        data[cname][0] = str(s_dt.date())  # Beginning Date
        data[cname][1] = str(t_dt.date())  # Trough Date
        data[cname][2] = f"{mdd:.1%}"      # Rep. Endowment MaxDD
        len_q = approximate_quarters_diff(s_dt, t_dt)
        data[cname][3] = f"{len_q}"
        if (s_dt in rep_cum.index) and (t_dt in rep_cum.index):
            r_ttr = find_time_to_recovery_q(rep_cum, s_dt, t_dt)
            data[cname][4] = f"{r_ttr}" if r_ttr else "n/a"
        else:
            data[cname][4] = "n/a"

        # row 5 => midrule1
        data[cname][5] = "midrule1"

        # rows 6,7 => Public Equity Drawdown + Time to Recovery
        if pe_cum is not None and not pe_cum.empty \
           and s_dt in pe_cum.index and t_dt in pe_cum.index:
            pe_pk = pe_cum.loc[s_dt]
            pe_th = pe_cum.loc[t_dt]
            eq_dd = (pe_th - pe_pk) / pe_pk
            data[cname][6] = f"{eq_dd:.1%}" if eq_dd < -1e-9 else "n/a"

            eq_ttr = find_time_to_recovery_q(pe_cum, s_dt, t_dt)
            data[cname][7] = f"{eq_ttr}" if eq_ttr else "n/a"
        else:
            data[cname][6] = "n/a"
            data[cname][7] = "n/a"

        # row 8 => midrule2
        data[cname][8] = "midrule2"

        # Hedge strategies
        offset = len(row_labels)  # first row index for hedges
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset + idx_h
            strat_cum = cum_dict[strat_name]
            # measure cumulative performance from start to end
            if s_dt in strat_cum.index and e_dt in strat_cum.index:
                start_val = strat_cum.loc[s_dt]
                end_val   = strat_cum.loc[e_dt]
                if start_val <= 1e-9:
                    data[cname][row_idx] = "n/a"
                else:
                    perf = (end_val / start_val) - 1.0
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
        r"\footnotesize{Crises identified at 5\% threshold on Representative Endowment. "
        r"Public Equity lines are divided by midrules. Hedging strategies show cumulative "
        r"performance from crisis start to end.}}"
    )
    lines.append(r"\label{table:hedge_crisis}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for row_lbl in row_labels:
        # If we see 'midrule1' or 'midrule2', we do a midrule and skip that row label
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
#    NEW: PIVOT TABLE WITH 90/10 OVERLAY ROWS FOR EACH HEDGING STRATEGY
###############################################################################
def pivot_crises_quarterly_expanded(rep_crises, cum_dict):
    """
    Build a pivot table for quarterly crises structured in three blocks:
      Block 1: Representative Endowment drawdown characteristics.
      Block 2: Hedge Strategy (Peak-to-Trough Returns).
      Block 3: Rep. Endowment + 10% Overlay Performance.
    For blocks 2 and 3, the performance is computed only over the peak-to-trough period 
    (from the crisis 'Start' to the 'Trough').
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    # Block 1: Representative Endowment drawdowns
    base_rows = [
        "Peak Date",
        "Trough Date",
        "Max Drawdown",
        "Drawdown Length (Qtrs)",
        "Time to Recovery (Qtrs)",
        "midrule1"  # Marker for block divider
    ]

    # Identify hedging columns (exclude Representative Endowment and Public Equity)
    exclude_labels = ["Representative Endowment", "Public Equity"]
    hedge_strategies = [
        k for k in cum_dict.keys() if k not in exclude_labels and not cum_dict[k].empty
    ]

    # Block 2: Hedge Strategy performance (one row per hedge strategy)
    block2_rows = hedge_strategies

    # Block 3: 90/10 Overlay performance (one row per hedge strategy)
    block3_rows = [f"Rep. Endowment + 10\\% {strat}" for strat in hedge_strategies]

    final_rows = base_rows + block2_rows + ["midrule2"] + block3_rows
    data = {c: [""] * len(final_rows) for c in col_names}

    def approximate_quarters_diff(ts_start, ts_end):
        days_diff = (ts_end - ts_start).days
        if days_diff < 0:
            return 0
        months = days_diff / 30.4375
        return max(1, int(math.ceil(months / 3)))

    def find_time_to_recovery_q(cum_series, start_d, trough_d):
        if start_d not in cum_series.index or trough_d not in cum_series.index:
            return None
        peak_val = cum_series.loc[start_d]
        subset = cum_series.loc[trough_d:]
        rec_idx = subset[subset >= peak_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_quarters_diff(trough_d, rec_idx[0])
    
    def compute_overlay_performance(rep_series, hedge_series, s_dt, t_dt, weight=0.1):
        """
        Compute the performance of a portfolio rebalanced quarterly as:
          90% Representative Endowment + 10% Hedge Strategy.
        For the subperiod from s_dt (peak) to t_dt (trough), this function computes the quarterly returns
        for each series, then combines them using the specified weights, and finally recumulated.
        """
        if s_dt not in rep_series.index or t_dt not in rep_series.index \
           or s_dt not in hedge_series.index or t_dt not in hedge_series.index:
            return None
        rep_sub = rep_series.loc[s_dt:t_dt]
        hedge_sub = hedge_series.loc[s_dt:t_dt]
        # If the subperiod has only one observation, compute a direct return
        if len(rep_sub) < 2 or len(hedge_sub) < 2:
            rep_return = rep_series.loc[t_dt] / rep_series.loc[s_dt] - 1
            hedge_return = hedge_series.loc[t_dt] / hedge_series.loc[s_dt] - 1
            return (1 - weight) * rep_return + weight * hedge_return
        # Compute quarterly returns
        rep_ret = rep_sub.pct_change().dropna()
        hedge_ret = hedge_sub.pct_change().dropna()
        df = pd.DataFrame({"rep": rep_ret, "hedge": hedge_ret}).dropna()
        if df.empty:
            return None
        combined_ret = (1 - weight) * df["rep"] + weight * df["hedge"]
        cum = (1 + combined_ret).cumprod()
        return cum.iloc[-1] - 1

    rep_cum = cum_dict.get("Representative Endowment", pd.Series(dtype=float))

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        s_dt = crisis['Start']
        t_dt = crisis['Trough']  # Use trough date for performance calculation
        mdd = crisis['Max Drawdown']

        # Block 1: Representative Endowment drawdowns
        data[cname][0] = str(s_dt.date())  # Peak Date
        data[cname][1] = str(t_dt.date())    # Trough Date
        data[cname][2] = f"{mdd:.1%}"         # Max Drawdown
        len_q = approximate_quarters_diff(s_dt, t_dt)
        data[cname][3] = f"{len_q}"
        r_ttr = find_time_to_recovery_q(rep_cum, s_dt, t_dt) if (s_dt in rep_cum.index and t_dt in rep_cum.index) else None
        data[cname][4] = f"{r_ttr}" if r_ttr else "n/a"

        # Block 2: Hedge Strategy performance (peak-to-trough)
        offset_block2 = len(base_rows)
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset_block2 + idx_h
            strat_cum = cum_dict[strat_name]
            if s_dt in strat_cum.index and t_dt in strat_cum.index:
                start_val = strat_cum.loc[s_dt]
                end_val   = strat_cum.loc[t_dt]
                if start_val <= 1e-9:
                    data[cname][row_idx] = "n/a"
                else:
                    perf = (end_val / start_val) - 1.0
                    data[cname][row_idx] = f"{perf:.1%}"
            else:
                data[cname][row_idx] = "n/a"

        # Block 3: 90/10 Overlay performance (peak-to-trough, rebalanced quarterly)
        offset_block3 = len(base_rows) + len(hedge_strategies) + 1  # +1 for midrule2 marker
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_idx = offset_block3 + idx_h
            overlay_perf = compute_overlay_performance(rep_cum, cum_dict[strat_name], s_dt, t_dt, weight=0.1)
            if overlay_perf is None:
                data[cname][row_idx] = "n/a"
            else:
                data[cname][row_idx] = f"{overlay_perf:.1%}"

    return pd.DataFrame(data, index=final_rows)

###############################################################################
#     EXPORTER FOR THE EXPANDED TABLE
###############################################################################
def export_table_to_latex_pivot_expanded(df, description="Quarterly Crisis Table: Drawdowns, Hedge Strategy, and 90/10 Overlay Performance"):
    """
    Convert the pivot table to LaTeX code formatted for the Journal of Finance.
    The table is structured in three blocks divided by midrules:
      • Block 1: Representative Endowment drawdowns,
      • Block 2: Hedge Strategy (Peak-to-Trough Returns),
      • Block 3: Rep. Endowment + 10% Overlay Performance.
    """
    columns = df.columns.tolist()
    align_str = "l" + "".join(["c"] * len(columns))

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\"  
        r"\footnotesize{Major crisis periods are reported along with the drawdown characteristics of the Representative Endowment. "
        r"Hedge strategy peak-to-trough returns and their corresponding 90/10 overlay performances are also presented.}}"
    )
    lines.append(r"\label{table:hedge_crisis_jf}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header for crisis columns
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Insert Block 1 header
    lines.append(r"\textbf{Representative Endowment} \\")
    
    # Iterate over the pivot table rows
    for row_lbl in df.index:
        if row_lbl == "midrule1":
            lines.append(r"\midrule")
            lines.append(r"\textbf{Hedge Strategy (Peak-to-Trough Returns)} \\")
            continue
        if row_lbl == "midrule2":
            lines.append(r"\midrule")
            lines.append(r"\textbf{Rep. Endowment + 10\% Overlay Performance} \\")
            continue

        # For detail rows, indent the row label
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
#                JF-STYLE CHART
###############################################################################
def create_performance_chart_jof_matplotlib(cum_dict, dd_dict):
    """
    2-row figure: top=cumulative, bottom=drawdowns, grayscale style.
    We'll fix legend so it doesn't go too far below.

    Also ensure TSM lines are all dashed or somewhat consistent.
    We'll do: if "Time Series Momentum" in label => dashed
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,5))

    def is_tsm_strategy(lbl: str) -> bool:
        return "Time Series Momentum" in lbl

    label_map = {}
    for orig_label in cum_dict.keys():
        if orig_label == "Representative Endowment":
            label_map[orig_label] = "Rep. Endowment"
        else:
            label_map[orig_label] = orig_label

    tsm_colors = [
        "dimgray", "gray", "darkgray", "silver", "lightgray"
    ]
    tsm_color_index = 0

    def style_for_label(lbl: str):
        nonlocal tsm_color_index
        if lbl == "Rep. Endowment":
            return dict(color='black', linestyle='solid', linewidth=1.2)
        elif lbl == "Public Equity":
            return dict(color='darkgray', linestyle='solid', linewidth=1.2)
        elif is_tsm_strategy(lbl):
            chosen_color = tsm_colors[tsm_color_index % len(tsm_colors)]
            tsm_color_index += 1
            return dict(color=chosen_color, linestyle='dashed', linewidth=1.2)
        else:
            return dict(color='dimgray', linestyle='solid', linewidth=1.2)

    # top: cumulative
    for lbl, series_c in cum_dict.items():
        if series_c.empty:
            continue
        style_ = style_for_label(label_map[lbl])
        plt_series = series_c.dropna()
        if not plt_series.empty:
            ax1.plot(
                plt_series.index, plt_series.values,
                label=label_map[lbl],
                **style_
            )

    ax1.set_ylabel("Cumulative Return", fontsize=8)
    ax1.tick_params(labelsize=8)

    # bottom: drawdown
    tsm_color_index = 0
    for lbl, series_c in dd_dict.items():
        if series_c.empty:
            continue
        style_ = style_for_label(label_map[lbl])
        plt_series = series_c.dropna()
        if not plt_series.empty:
            ax2.plot(
                plt_series.index, plt_series.values,
                label=label_map[lbl],
                **style_
            )

    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.tick_params(labelsize=8)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    handles_top, labels_top = ax1.get_legend_handles_labels()
    handles_bot, labels_bot = ax2.get_legend_handles_labels()
    all_handles = handles_top + handles_bot
    all_labels = labels_top + labels_bot

    seen = {}
    dedup_handles = []
    dedup_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            dedup_handles.append(h)
            dedup_labels.append(l)
            seen[l] = True

    fig.legend(
        dedup_handles, dedup_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=8
    )

    fig.tight_layout(pad=1.5)
    plt.subplots_adjust(bottom=0.20)
    return fig


###############################################################################
#                              MAIN APP
###############################################################################
def main():
    st.title("Quarterly Crisis Period Table + Cumulative Hedge Performance")

    # 1) Load main data
    allocations, ret_q_full = load_and_preprocess_data()
    # Use a copy of ret_q_full to create the unified timeframe for allocations-based returns
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q_full.copy())
    idx_q = pd.date_range(q_start, q_end, freq='Q-DEC')

    # Build Rep. Endowment (Q) using the unified timeframe
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)
    valid_cols = hist_alloc_q.columns.intersection(ret_q.columns)
    hist_alloc_q, ret_q_hist = hist_alloc_q[valid_cols], ret_q[valid_cols]
    rep_endw_q = (hist_alloc_q * ret_q_hist).sum(axis=1).dropna()
    rep_cum, rep_dd = calculate_cumulative_and_dd(rep_endw_q)

    # 2) Load hedging strategies (monthly) => unify => resample to Q => build cumulative
    hedge_df = load_hedging_strategies()
    hedge_df = hedge_df[(hedge_df.index >= q_start) & (hedge_df.index <= q_end)]
    # resample to quarterly, picking last monthly data in each quarter
    hedge_q = hedge_df.resample('Q').last().dropna(how='all')

    # Build cum_dict for original pivot
    cum_dict = {"Representative Endowment": rep_cum}
    # For Public Equity, use the full returns data (ret_q_full) so it always includes data since 2000.
    if "Public Equity" in ret_q_full.columns:
        pe_ = ret_q_full["Public Equity"].dropna()
        pe_cum, _ = calculate_cumulative_and_dd(pe_)
        cum_dict["Public Equity"] = pe_cum

    for col in hedge_q.columns:
        col_series = hedge_q[col].dropna()
        if not col_series.empty:
            c_, _ = calculate_cumulative_and_dd(col_series)
            cum_dict[col] = c_

    # 3) Identify crises using Rep. Endowment’s drawdowns
    crises = find_crisis_periods_for_rep_endowment(rep_dd, threshold=0.05)

    # (Rest of your code remains unchanged...)
    # -------------------------------------------------------------------------
    # 4) Original pivot table (unchanged)
    pivoted = pivot_crises_quarterly(crises, cum_dict)
    st.subheader("Quarterly Crisis Table with Hedging Strategies: Cumulative Performance (Original)")
    if pivoted.empty:
        st.warning("No crises found or no data to display.")
    else:
        st.dataframe(pivoted)
        latex_code = export_table_to_latex_pivot(
            pivoted,
            description="Quarterly Crisis Periods + Hedge Strategy Cumulative Performance"
        )
        st.write("**LaTeX for the Crisis + Hedge Performance Table (Original):**")
        st.code(latex_code, language="latex")

    # -------------------------------------------------------------------------
    # 5) Expanded pivot table with 90/10 rows
    expanded_pivot = pivot_crises_quarterly_expanded(crises, cum_dict)
    st.subheader("Quarterly Crisis Table with Hedging Strategies + 90/10 Overlays")
    if expanded_pivot.empty:
        st.warning("No crises found or no data to display (expanded).")
    else:
        st.dataframe(expanded_pivot)
        latex_expanded = export_table_to_latex_pivot_expanded(
            expanded_pivot,
            description="Quarterly Crisis Periods + Hedge Strategy Cumulative Performance (90/10 Overlays)"
        )
        st.write("**LaTeX for the Crisis + Hedge Performance Table (Expanded 90/10):**")
        st.code(latex_expanded, language="latex")

    # -------------------------------------------------------------------------
    # 6) JF-style Chart
    dd_dict = {}
    for lbl, series_c in cum_dict.items():
        if series_c.empty:
            dd_dict[lbl] = pd.Series(dtype=float)
        else:
            run_max = series_c.cummax()
            dd_ = (series_c - run_max) / run_max
            dd_dict[lbl] = dd_
    st.subheader("Cumulative & Drawdown Chart (JF-Style)")
    fig = create_performance_chart_jof_matplotlib(cum_dict, dd_dict)
    st.pyplot(fig)

    st.write("Above chart compares the Representative Endowment, Public Equity, "
             "and each hedging strategy on a cumulative-return and drawdown basis, "
             "sampled quarterly. Time Series Momentum lines are dashed and share "
             "a similar style. The legend is placed just below the x-axis labels "
             "to avoid overlap.")


if __name__ == "__main__":
    main()
