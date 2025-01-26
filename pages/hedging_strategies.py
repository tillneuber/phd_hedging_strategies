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
    return df


###############################################################################
#               CRISIS DETECTION (REP ENDOWMENT)
###############################################################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.05):
    """
    Identify crisis periods in the Representative Endowment's drawdown if ≥ threshold (5%).
    Return list of dicts: { 'Start', 'Trough', 'End', 'Max Drawdown' }.
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
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
                if max_dd <= -threshold:
                    crises.append({
                        'Start': start_date,
                        'Trough': trough_date,
                        'End': end_date,
                        'Max Drawdown': max_dd
                    })
                start_date, trough_date, max_dd = None, None, 0.0

    if in_crisis:
        end_date = ds.index[-1]
        if max_dd <= -threshold:
            crises.append({
                'Start': start_date,
                'Trough': trough_date,
                'End': end_date,
                'Max Drawdown': max_dd
            })
    return crises


###############################################################################
#           PIVOT TABLE: SHOW HEDGING STRATEGIES' CUMULATIVE PERF
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
      6) ---
      7) Public Equity Drawdown
      8) Public Equity Time to Recovery
      9) ---
      For each hedging strategy => "HedgeName Cumulative Performance (Start->End)"

    We do NOT show the drawdown of the hedging strategies; 
    we show their *cumulative performance* from crisis start to crisis end.
    And we remove "time to recovery" for hedges, as requested.
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    base_rows = [
        "Beginning Date",
        "Trough Date",
        "Rep. Endowment Max Drawdown",
        "Drawdown Length (quarters)",
        "Time to Recovery (quarters)",
        "---",
        "Public Equity Drawdown",
        "Public Equity Time to Recovery",
        "---2"
    ]

    # Identify all "hedging" columns except "Representative Endowment" and "Public Equity"
    exclude_labels = ["Representative Endowment", "Public Equity"]
    # We'll skip any short, empty series
    hedge_strategies = [k for k in cum_dict.keys()
                        if k not in exclude_labels and not cum_dict[k].empty]

    # Each hedge strategy => single row "Strategy Performance Over Crisis"
    row_entries = []
    for strat in hedge_strategies:
        row_entries.append(f"{strat} (Start→End Perf)")

    row_labels = base_rows + row_entries
    data = {c: [""] * len(row_labels) for c in col_names}

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

    # Quick references:
    rep_cum = cum_dict.get("Representative Endowment", pd.Series(dtype=float))
    pe_cum = cum_dict.get("Public Equity", pd.Series(dtype=float))

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        s_dt = crisis['Start']
        t_dt = crisis['Trough']
        e_dt = crisis['End']
        mdd = crisis['Max Drawdown']

        # Row 0: Beginning Date
        data[cname][0] = str(s_dt.date())
        # Row 1: Trough Date
        data[cname][1] = str(t_dt.date())
        # Row 2: Rep. Endowment Max Drawdown
        data[cname][2] = f"{mdd:.1%}"

        # Row 3: Drawdown Length
        len_q = approximate_quarters_diff(s_dt, t_dt)
        data[cname][3] = f"{len_q} q"

        # Row 4: Time to Recovery
        if not rep_cum.empty and (s_dt in rep_cum.index) and (t_dt in rep_cum.index):
            r_ttr = find_time_to_recovery_q(rep_cum, s_dt, t_dt)
            data[cname][4] = f"{r_ttr} q" if r_ttr else "n/a"
        else:
            data[cname][4] = "n/a"

        # Row 5: --- (midrule)
        data[cname][5] = "---"

        # Row 6: Public Equity Drawdown
        # Row 7: Public Equity Time to Recovery
        if pe_cum is not None and not pe_cum.empty \
           and s_dt in pe_cum.index and t_dt in pe_cum.index:
            pk_val = pe_cum.loc[s_dt]
            th_val = pe_cum.loc[t_dt]
            eq_dd = (th_val - pk_val)/pk_val
            data[cname][6] = f"{eq_dd:.1%}" if eq_dd < -1e-9 else "n/a"
            eq_ttr = find_time_to_recovery_q(pe_cum, s_dt, t_dt)
            data[cname][7] = f"{eq_ttr} q" if eq_ttr else "n/a"
        else:
            data[cname][6] = "n/a"
            data[cname][7] = "n/a"

        # Row 8: ---2
        data[cname][8] = "---2"

        # Then the hedge strategies
        offset = len(base_rows)
        # For each hedge strategy => single row: cumulative performance from Start -> End
        for idx_h, strat_name in enumerate(hedge_strategies):
            row_index = offset + idx_h
            # s_dt => e_dt => performance
            strat_cum = cum_dict[strat_name]
            if s_dt in strat_cum.index and e_dt in strat_cum.index:
                start_val = strat_cum.loc[s_dt]
                end_val = strat_cum.loc[e_dt]
                if start_val <= 1e-9:
                    data[cname][row_index] = "n/a"
                else:
                    perf = (end_val / start_val) - 1.0
                    data[cname][row_index] = f"{perf:.1%}"
            else:
                data[cname][row_index] = "n/a"

    return pd.DataFrame(data, index=row_labels)


def export_table_to_latex_pivot(df, description="Quarterly Crisis Table + Hedging Perf"):
    """
    Convert pivot table to LaTeX code. 
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
        r"Hedging strategies show cumulative performance (start→end).}}"
    )
    lines.append(r"\label{table:hedge_crisis}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for i, row_lbl in enumerate(row_labels):
        row_vals = [row_lbl]
        # fetch for each col
        for col in columns:
            val = df.loc[row_lbl, col]
            if isinstance(val, str):
                val = val.replace("%", r"\%")
            row_vals.append(val)
        line = " & ".join(row_vals) + r" \\"
        lines.append(line)
        # optional midrule if row contains '---'
        if i < len(row_labels) - 1 and '---' in row_lbl:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


###############################################################################
#                JF-STYLE CHART (IF NEEDED)
###############################################################################
def create_performance_chart_jof_matplotlib(cum_dict, dd_dict):
    """
    2-row figure: top = cumulative, bottom = drawdowns, grayscale style.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,4.6))

    # Label map
    label_map = {}
    for orig_label in cum_dict.keys():
        if orig_label == "Representative Endowment":
            label_map[orig_label] = "Rep. Endowment"
        else:
            label_map[orig_label] = orig_label

    # some line styles
    line_styles = [
        {'color': 'black',   'linestyle': 'solid'},
        {'color': 'darkgray','linestyle': 'solid'},
        {'color': 'dimgray', 'linestyle': 'dashed'},
        {'color': 'gray',    'linestyle': 'dotted'},
        {'color': 'silver',  'linestyle': 'dashdot'},
        {'color': 'lightgray','linestyle': 'solid'},
    ]

    def get_style(i):
        return line_styles[i % len(line_styles)]

    # TOP: cumulative
    i = 0
    for lbl, series_c in cum_dict.items():
        if series_c.empty:
            continue
        style_ = get_style(i)
        i += 1
        ax1.plot(
            series_c.index, series_c.values,
            label=label_map[lbl],
            color=style_['color'],
            linestyle=style_['linestyle'],
            linewidth=1.2
        )

    ax1.set_ylabel("Cumulative Return", fontsize=8)
    ax1.tick_params(labelsize=8)

    # BOTTOM: drawdown
    i = 0
    for lbl, series_c in dd_dict.items():
        if series_c.empty:
            continue
        style_ = get_style(i)
        i += 1
        ax2.plot(
            series_c.index, series_c.values,
            label=label_map[lbl],
            color=style_['color'],
            linestyle=style_['linestyle'],
            linewidth=1.2
        )

    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.tick_params(labelsize=8)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    # deduplicate legend
    handles_top, labels_top = ax1.get_legend_handles_labels()
    handles_bot, labels_bot = ax2.get_legend_handles_labels()
    all_handles = handles_top + handles_bot
    all_labels = labels_top + labels_bot

    unique = {}
    for h, l in zip(all_handles, all_labels):
        if l not in unique:
            unique[l] = h
    dedup_handles = list(unique.values())
    dedup_labels = list(unique.keys())

    fig.legend(
        dedup_handles, dedup_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        fontsize=8
    )

    fig.tight_layout(pad=1.2)
    return fig


###############################################################################
#                              MAIN APP
###############################################################################
def main():
    st.title("Quarterly Crisis Period Table + Cumulative Hedge Performance")

    # 1) Load main data
    allocations, ret_q = load_and_preprocess_data()
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q)
    idx_q = pd.date_range(q_start, q_end, freq='Q-DEC')

    # Build Rep. Endowment (Q)
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)
    valid_cols = hist_alloc_q.columns.intersection(ret_q.columns)
    hist_alloc_q, ret_q_hist = hist_alloc_q[valid_cols], ret_q[valid_cols]

    rep_endw_q = (hist_alloc_q * ret_q_hist).sum(axis=1).dropna()
    rep_cum, rep_dd = calculate_cumulative_and_dd(rep_endw_q)

    # 2) Load hedging strategies (monthly) => unify => resample to Q => build cumulative
    hedge_df = load_hedging_strategies()
    hedge_df = hedge_df[(hedge_df.index >= q_start) & (hedge_df.index <= q_end)]
    # resample to quarterly
    hedge_q = hedge_df.resample('Q').last().dropna(how='all')

    # Build cum_dict
    cum_dict = {"Representative Endowment": rep_cum}
    # see if Public Equity is in ret_q
    if "Public Equity" in ret_q.columns:
        pe_ = ret_q["Public Equity"].dropna()
        pe_cum, pe_dd = calculate_cumulative_and_dd(pe_)
        cum_dict["Public Equity"] = pe_cum

    # add each hedge column
    for col in hedge_q.columns:
        series = hedge_q[col].dropna()
        if not series.empty:
            c_, _ = calculate_cumulative_and_dd(series)
            cum_dict[col] = c_

    # 3) Identify crises
    crises = find_crisis_periods_for_rep_endowment(rep_dd, threshold=0.05)

    # 4) Pivot table
    pivoted = pivot_crises_quarterly(crises, cum_dict)

    st.subheader("Quarterly Crisis Table with Hedging Strategies' Cumulative Performance")
    if pivoted.empty:
        st.warning("No crises found or no data to display.")
    else:
        st.dataframe(pivoted)

        # LaTeX
        latex_code = export_table_to_latex_pivot(
            pivoted,
            description="Quarterly Crisis Periods + Hedge Strategy Performance"
        )
        st.write("**LaTeX for the Crisis + Hedge Performance Table:**")
        st.code(latex_code, language="latex")

    # 5) (Optional) Chart
    # Build dd_dict
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

    st.write("Above chart shows the Representative Endowment, Public Equity, "
             "and each hedging strategy's cumulative and drawdown paths on a quarterly timeline.")


if __name__ == "__main__":
    main()
