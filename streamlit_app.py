import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import math

########################################
#  FIX: DEFINE load_historical_saa()
########################################
def load_historical_saa():
    """
    Provide a dummy or alias function to avoid the NameError.
    Simply calls load_and_preprocess_data().
    """
    return load_and_preprocess_data()

########################################
#         DATA LOADING / PREP
########################################
def load_and_preprocess_data():
    """
    Load CSVs, parse dates, convert % to decimals, return DataFrames:
      1) allocations (historical endowment SAA)
      2) quarterly returns for various asset classes
    We add a 0.0% baseline row at 01-Jul-1999 for the quarterly data.
    """

    ########################################
    # Load the allocations
    ########################################
    alloc_cols = [
        'Year', 'Public Equity', 'PE/VC', 'Hedge Funds',
        'Real Assets & ILBs', 'Fixed Income', 'Private Credit', 'Cash'
    ]
    allocations = pd.read_csv('data/hist_endowment_saa.csv', sep=';', names=alloc_cols, header=0)

    # Convert wide->long
    allocations = allocations.melt(
        id_vars=['Year'], var_name='Asset Class', value_name='Allocation'
    )
    allocations['Allocation'] = pd.to_numeric(allocations['Allocation'], errors='coerce')
    allocations['Start Date'] = pd.to_datetime(allocations['Year'].astype(str)) + pd.DateOffset(months=6)
    allocations['End Date'] = allocations['Start Date'] + pd.DateOffset(years=1) - pd.DateOffset(days=1)

    ########################################
    # Load quarterly returns
    ########################################
    returns_quarterly = pd.read_csv('data/quarterly_returns.csv', sep=';', header=0)
    # Parse date
    returns_quarterly['Date'] = pd.to_datetime(returns_quarterly['Date'], format='%d.%m.%Y', errors='coerce')
    # Set as index
    returns_quarterly.set_index('Date', inplace=True)

    # Shift index to last day of that same month (if not already)
    # e.g. 30-Sep-1999 stays 30-Sep-1999, 31-Dec-1999 stays 31-Dec-1999, etc.
    returns_quarterly.index = returns_quarterly.index + pd.offsets.MonthEnd(0)

    # Convert string percentages to decimal
    returns_quarterly = returns_quarterly.apply(
        lambda col: col.map(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) else np.nan
        )
    )

    # ---- Add a zero-return baseline row at 01-Jul-1999 ----
    base_dt = pd.Timestamp("1999-07-01")
    if base_dt not in returns_quarterly.index:
        zero_vals = {col: 0.0 for col in returns_quarterly.columns}
        returns_quarterly.loc[base_dt] = zero_vals
    returns_quarterly.sort_index(inplace=True)

    return allocations, returns_quarterly


def load_individual_endowments():
    """
    Load CSV with columns like:
       Date;Yale;Stanford;Harvard;Average Endowment (NACUBO)
    Convert % -> decimals, keep the date as end-of-period,
    and add a 0.0% baseline row at 01-Jul-1999 for the annual data as well.
    """
    df = pd.read_csv('data/individual_endowments.csv', sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Convert string percentages to decimal
    for col in df.columns:
        df[col] = df[col].apply(lambda x: float(str(x).replace('%', ''))/100 if pd.notnull(x) else np.nan)

    # ---- Add a zero-return baseline row at 01-Jul-1999 ----
    base_dt = pd.Timestamp("1999-07-01")
    if base_dt not in df.index:
        zero_values = {col: 0.0 for col in df.columns}
        df.loc[base_dt] = zero_values
    df.sort_index(inplace=True)

    return df


def unify_timeframe(allocations, returns_q):
    """
    Align the timeframe for allocations & returns.
    """
    earliest_alloc_start = allocations['Start Date'].min()
    latest_alloc_end = allocations['End Date'].max()
    earliest_ret_q = returns_q.index.min()
    latest_ret_q = returns_q.index.max()

    q_start = max(earliest_alloc_start, earliest_ret_q)
    q_end = min(latest_alloc_end, latest_ret_q)

    # Filter allocations
    mask_alloc = (allocations['End Date'] >= q_start) & (allocations['Start Date'] <= q_end)
    allocations = allocations[mask_alloc]

    # Filter returns
    mask_rets = (returns_q.index >= q_start) & (returns_q.index <= q_end)
    returns_q = returns_q[mask_rets]

    return allocations, returns_q, q_start, q_end


def map_allocations_to_periods(allocations_df, date_index):
    """
    For each date in date_index, find the 'Allocation' within (Start Date->End Date).
    """
    allocations_mapped = pd.DataFrame(index=date_index)
    for asset in allocations_df['Asset Class'].unique():
        asset_df = allocations_df[allocations_df['Asset Class'] == asset]
        for _, row in asset_df.iterrows():
            period = pd.date_range(start=row['Start Date'], end=row['End Date'], freq=date_index.freq)
            allocations_mapped.loc[allocations_mapped.index.isin(period), asset] = row['Allocation']
    return allocations_mapped.fillna(0)


def calculate_cumulative_and_dd(returns_series):
    """
    Return (cumulative returns, drawdowns).
    """
    cumret = (1 + returns_series).cumprod()
    run_max = cumret.cummax()
    dd = (cumret - run_max) / run_max
    return cumret, dd


########################################
#  MATPLOTLIB JOIF-STYLE PERFORMANCE CHART
########################################
def create_performance_chart_jof_matplotlib(cum_dict, dd_dict):
    """
    Two-subplot figure: top=cumulative, bottom=drawdowns,
    grayscale style suitable for JoF.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,4.6))

    # Map or rename
    label_map = {}
    for orig_label in cum_dict.keys():
        # Example: remove "Latest SAA"
        if "Latest SAA" in orig_label:
            label_map[orig_label] = None
        elif ("Historical Endowment" in orig_label) or ("Representative Endowment" in orig_label):
            label_map[orig_label] = "Representative Endowment"
        else:
            clean_label = orig_label.replace("(Q)", "").replace("(Annual)", "").strip()
            label_map[orig_label] = clean_label

    line_styles = [
        {'color': 'black', 'linestyle': 'solid'},
        {'color': 'darkgray', 'linestyle': 'solid'},
        {'color': 'dimgray', 'linestyle': 'dashed'},
        {'color': 'gray', 'linestyle': 'dotted'},
        {'color': 'silver', 'linestyle': 'dashdot'},
        {'color': 'lightgray', 'linestyle': 'solid'},
    ]

    def get_style(i):
        return line_styles[i % len(line_styles)]

    def choose_line_style(label_cleaned, default_style):
        if label_cleaned == "Representative Endowment":
            return {'color': 'darkgray', 'linestyle': 'solid', 'linewidth':1.2}
        if label_cleaned == "Public Equity":
            return {'color': 'black', 'linestyle': 'solid', 'linewidth':1.2}
        if "Stanford" in label_cleaned:
            return {'color': 'black', 'linestyle': 'dotted', 'linewidth':1.2}
        if "Average Endowment" in label_cleaned:
            return {'color': 'dimgray', 'linestyle': 'dotted', 'linewidth':1.2}
        return default_style

    # Top Subplot => Cumulative
    i = 0
    for label, series in cum_dict.items():
        final_lbl = label_map.get(label, None)
        if final_lbl is None or series.empty:
            continue
        style_candidate = get_style(i)
        i += 1
        final_style = choose_line_style(final_lbl, style_candidate)
        plt_series = series.dropna()
        if plt_series.empty:
            continue

        ax1.plot(plt_series.index, plt_series.values,
                 label=final_lbl,
                 color=final_style['color'],
                 linestyle=final_style['linestyle'],
                 linewidth=final_style.get('linewidth', 1.2))

    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel("Cumulative Returns", fontsize=8)
    ax1.tick_params(labelsize=8)

    # Bottom Subplot => Drawdowns
    i = 0
    for label, series in dd_dict.items():
        final_lbl = label_map.get(label, None)
        if final_lbl is None or series.empty:
            continue
        style_candidate = get_style(i)
        i += 1
        final_style = choose_line_style(final_lbl, style_candidate)
        plt_series = series.dropna()
        if plt_series.empty:
            continue

        ax2.plot(plt_series.index, plt_series.values,
                 label=final_lbl,
                 color=final_style['color'],
                 linestyle=final_style['linestyle'],
                 linewidth=final_style.get('linewidth', 1.2))

    ax2.set_ylabel("Drawdowns", fontsize=8)
    ax2.tick_params(labelsize=8)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Deduplicate legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
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


########################################
#  CRISIS DETECTION (≥ 5%) QUARTERLY
########################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.05):
    """
    Identify crisis periods in the Representative Endowment's quarterly drawdown if ≥ threshold (5%).
    - Peak Date: The last date before the drawdown starts (i.e. when the drawdown was zero or positive).
    - Trough Date: The date when the maximum drawdown is reached.
    - Crises where the peak and trough dates are identical are skipped.
    
    Returns a list of dictionaries with keys: 'Start', 'Trough', 'End', 'Max Drawdown'.
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
            # Set peak date as the previous date if its drawdown was non-negative
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


########################################
#    CRISIS DETECTION (≥ 5%) ANNUAL NACUBO
########################################
def find_crisis_periods_for_nacubo_annual(returns_series, threshold=0.05):
    """
    Identify crisis periods in the Average Endowment (NACUBO) annual data if
    the peak-to-trough decline ≥ threshold (5%).
    - Peak Date: The last date before the drawdown begins (local maximum).
    - Trough Date: The date when the maximum drawdown is reached.
    - Crises with identical peak and trough dates are skipped.
    
    Returns a list of dictionaries: 'Start', 'Trough', 'End', 'Max Drawdown'.
    """
    ds = returns_series.dropna().copy()
    if ds.empty:
        return []
    cumret = (1 + ds).cumprod()
    run_max = cumret.cummax()
    dd = (cumret - run_max) / run_max

    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in dd.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = dd.index.get_loc(date)
            if idx > 0 and dd.iloc[idx - 1] >= 0:
                start_date = dd.index[idx - 1]
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
                if max_dd <= -threshold and start_date != trough_date:
                    crises.append({
                        'Start': start_date,
                        'Trough': trough_date,
                        'End': end_date,
                        'Max Drawdown': max_dd
                    })
                start_date, trough_date, max_dd = None, None, 0.0

    if in_crisis:
        end_date = dd.index[-1]
        if max_dd <= -threshold and start_date != trough_date:
            crises.append({
                'Start': start_date,
                'Trough': trough_date,
                'End': end_date,
                'Max Drawdown': max_dd
            })
    return crises


def approximate_years_diff(ts_start, ts_end):
    """
    Approximate difference in years by day difference / ~365.25
    """
    days_diff = (ts_end - ts_start).days
    return days_diff / 365.25


def find_time_to_recovery_annual(returns_series, start_date, trough_date):
    """
    For annual data. We calculate the time from trough_date until
    it recovers the cumulative value at start_date. Returns # of years or np.nan.
    """
    if start_date not in returns_series.index or trough_date not in returns_series.index:
        return np.nan

    cum_ret = (1 + returns_series).cumprod()
    start_val = cum_ret.loc[start_date]
    subset = cum_ret.loc[trough_date:]
    rec_idx = subset[subset >= start_val].index
    if len(rec_idx) == 0:
        return np.nan
    return approximate_years_diff(trough_date, rec_idx[0])


########################################
#  TABLE BUILDING - QUARTERLY
########################################
def pivot_crises_into_columns(rep_crises, cum_dict):
    """
    For quarterly-based crises from the Representative Endowment.
    Pivot columns => Crisis1, Crisis2, ...
    The table omits individual endowments from this table, only showing Public Equity.
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]
    row_labels = [
        "Beginning Date",
        "Trough Date",
        "Max Drawdown",
        "Drawdown Length (quarters)",
        "Time to Recovery (quarters)",
        "---",
        "Public Equity Drawdown",
        "Public Equity Time to Recovery"
    ]
    data = {c: [""] * len(row_labels) for c in col_names}

    def approximate_quarters_diff(ts_start, ts_end):
        # approximate months/3 => quarters
        days_diff = (ts_end - ts_start).days
        months = days_diff / 30.4375
        # round up to at least 1 if there's any partial
        q = math.ceil(months/3)
        return q if q>0 else 1

    def find_time_to_recovery_q(cum_series, start_d, trough_d):
        """
        Returns an integer # of quarters.
        """
        if start_d not in cum_series.index or trough_d not in cum_series.index:
            return None
        peak_val = cum_series.loc[start_d]
        subset = cum_series.loc[trough_d:]
        rec_idx = subset[subset >= peak_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_quarters_diff(trough_d, rec_idx[0])

    row_map = {
        "Public Equity": (6, 7)
    }

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        start_dt = crisis['Start']
        trough_dt = crisis['Trough']
        end_dt = crisis['End']
        max_dd = crisis['Max Drawdown']

        # length in quarters
        length_q = approximate_quarters_diff(start_dt, trough_dt)
        data[cname][0] = str(start_dt.date())
        data[cname][1] = str(trough_dt.date())
        data[cname][2] = f"{max_dd:.1%}"
        data[cname][3] = f"{length_q} q"

        # time to recovery
        rep_cum = cum_dict["Representative Endowment"]
        rep_ttr = find_time_to_recovery_q(rep_cum, start_dt, trough_dt)
        if rep_ttr is None:
            data[cname][4] = "n/a"
        else:
            data[cname][4] = f"{rep_ttr} q"

        # fill for Public Equity only
        for entity, (r_decline, r_recovery) in row_map.items():
            if entity not in cum_dict:
                continue
            e_cum = cum_dict[entity]
            if start_dt not in e_cum.index or trough_dt not in e_cum.index:
                data[cname][r_decline] = "n/a"
                data[cname][r_recovery] = "n/a"
                continue
            pk_val = e_cum.loc[start_dt]
            th_val = e_cum.loc[trough_dt]
            decline_e = (th_val - pk_val)/pk_val
            if decline_e > -1e-9:
                data[cname][r_decline] = "n/a"
            else:
                data[cname][r_decline] = f"{decline_e:.1%}"

            e_ttr_ = find_time_to_recovery_q(e_cum, start_dt, trough_dt)
            if e_ttr_ is None:
                data[cname][r_recovery] = "n/a"
            else:
                data[cname][r_recovery] = f"{e_ttr_} q"

    df_out = pd.DataFrame(data, index=row_labels)
    return df_out


########################################
#  TABLE BUILDING - ANNUAL NACUBO
########################################
def pivot_crises_into_columns_annual(nacubo_crises, annual_cum_dict):
    """
    For annual NACUBO-based crises. Columns => Crisis1, Crisis2, ...
    Show Public Equity, Rep Endowment, Yale,Stanford,Harvard.
    Time to Recovery => integer # of years.
    """
    if not nacubo_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(nacubo_crises))]
    row_labels = [
        "Beginning FY",
        "Trough FY",
        "Max Drawdown",
        "Drawdown Length (yrs)",
        "Time to Recovery (yrs)",
        "---",
        "Public Equity Drawdown",
        "Public Equity Time to Recovery",
        "---2",
        "Representative Endowment Drawdown",
        "Representative Endowment Time to Recovery",
        "---3",
        "Yale Drawdown",
        "Yale Time to Recovery",
        "Stanford Drawdown",
        "Stanford Time to Recovery",
        "Harvard Drawdown",
        "Harvard Time to Recovery"
    ]
    data = {c: [""] * len(row_labels) for c in col_names}

    def approximate_years_diff(ts_start, ts_end):
        days_diff = (ts_end - ts_start).days
        val = math.ceil(days_diff/365.25)
        return val if val>0 else 1

    def find_time_to_recovery_annual(cum_s, s_d, t_d):
        """
        Returns an integer # of years. 
        """
        if s_d not in cum_s.index or t_d not in cum_s.index:
            return None
        start_val = cum_s.loc[s_d]
        subset = cum_s.loc[t_d:]
        rec_idx = subset[subset >= start_val].index
        if len(rec_idx) == 0:
            return None
        return approximate_years_diff(t_d, rec_idx[0])

    row_map = {
        "Public Equity": (6, 7),
        "Representative Endowment": (9, 10),
        "Yale": (12, 13),
        "Stanford": (14, 15),
        "Harvard": (16, 17)
    }

    for i, crisis in enumerate(nacubo_crises):
        cname = col_names[i]
        start_dt = crisis['Start']
        trough_dt = crisis['Trough']
        end_dt = crisis['End']
        max_dd = crisis['Max Drawdown']

        data[cname][0] = str(start_dt.date())
        data[cname][1] = str(trough_dt.date())
        data[cname][2] = f"{max_dd:.1%}"

        days_diff = (trough_dt - start_dt).days
        yrs_len = math.ceil(days_diff/365.25)
        data[cname][3] = f"{yrs_len} yrs"

        # time to recovery for NACUBO
        if "Average Endowment (NACUBO)" in annual_cum_dict:
            nacubo_cum = annual_cum_dict["Average Endowment (NACUBO)"]
            if (start_dt in nacubo_cum.index) and (trough_dt in nacubo_cum.index):
                ttr_n = find_time_to_recovery_annual(nacubo_cum, start_dt, trough_dt)
                if ttr_n is None:
                    data[cname][4] = "n/a"
                else:
                    data[cname][4] = f"{ttr_n} yrs"
            else:
                data[cname][4] = "n/a"

        # fill for the row_map
        for entity, (r_decline, r_recovery) in row_map.items():
            if entity not in annual_cum_dict:
                data[cname][r_decline] = "n/a"
                data[cname][r_recovery] = "n/a"
                continue

            e_cum = annual_cum_dict[entity]
            if (start_dt not in e_cum.index) or (trough_dt not in e_cum.index):
                data[cname][r_decline] = "n/a"
                data[cname][r_recovery] = "n/a"
                continue

            pk_val = e_cum.loc[start_dt]
            th_val = e_cum.loc[trough_dt]
            decline_e = (th_val - pk_val)/pk_val
            if decline_e > -1e-9:
                data[cname][r_decline] = "n/a"
            else:
                data[cname][r_decline] = f"{decline_e:.1%}"

            e_ttr_ = find_time_to_recovery_annual(e_cum, start_dt, trough_dt)
            if e_ttr_ is None:
                data[cname][r_recovery] = "n/a"
            else:
                data[cname][r_recovery] = f"{e_ttr_} yrs"

    return pd.DataFrame(data, index=row_labels)


def export_table_to_latex_pivot(df, description=""):
    """
    Convert pivoted df to LaTeX code, 
    using an alignment that doesn't break with & in the preamble.
    """
    columns = df.columns.tolist()
    row_labels = df.index.tolist()

    # We'll do first column => l, then all columns => c
    align_str = "l" + "".join(["c"] * len(columns))

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\"
        r"\footnotesize{Crisis identification based on a 5\% drawdown threshold.}}"
    )
    lines.append(r"\label{table:crisis_pivot}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header line
    header = [""] + columns
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for idx, row_lbl in enumerate(row_labels):
        if row_lbl.startswith("---"):
            # divider
            lines.append(r"\midrule")
            continue

        rowvals = [row_lbl]
        for col in columns:
            val = df.loc[row_lbl, col]
            if isinstance(val, str):
                # Escape percent signs
                val = val.replace("%", r"\%")
            rowvals.append(str(val))
        line = " & ".join(rowvals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


########################################
#                MAIN
########################################
def main():
    st.title("Approach #1: Two Separate Tables for Quarterly vs. Annual Crises")

    # 1) Load Data
    allocations, ret_q_full = load_and_preprocess_data()
    # Create a unified timeframe for allocations-based calculations.
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q_full.copy())

    # NOTE: By default, let's pick a quarterly frequency that matches 'QE-DEC'
    idx_q = pd.date_range(q_start, q_end, freq='QE-DEC')

    # Calculate Representative Endowment returns using the unified timeframe.
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)
    common_cols = hist_alloc_q.columns.intersection(ret_q.columns)
    hist_alloc_q, ret_q_hist = hist_alloc_q[common_cols], ret_q[common_cols]
    hist_endw_returns = (hist_alloc_q * ret_q_hist).sum(axis=1)
    rep_cum, rep_dd = calculate_cumulative_and_dd(hist_endw_returns)

    # Build the dictionary for the chart (quarterly).
    cum_dict_quarterly = {
        "Representative Endowment": rep_cum
    }

    # For Public Equity, we can attach the full returns so it starts in 1999 as well
    if 'Public Equity' in ret_q_full.columns:
        pe_returns = ret_q_full['Public Equity'].dropna()
        pe_cum, pe_dd = calculate_cumulative_and_dd(pe_returns)
        cum_dict_quarterly["Public Equity"] = pe_cum

    # Now add the annual endowments too (Yale,Stanford,Harvard,NACUBO)
    endow_df = load_individual_endowments()
    for col in endow_df.columns:
        s_ = endow_df[col].dropna().sort_index()
        if not s_.empty:
            c_, d_ = calculate_cumulative_and_dd(s_)
            cum_dict_quarterly[col] = c_

    # ====== FIRST TABLE: QUARTERLY CRISES FOR REP. ENDOWMENT ======
    rep_crises_q = find_crisis_periods_for_rep_endowment(rep_dd, threshold=0.05)
    pivot_df_q = pivot_crises_into_columns(rep_crises_q, cum_dict_quarterly)

    st.subheader("Table 1: Quarterly Crisis Periods (≥5% Drawdown for Representative Endowment)")
    st.write("This table shows crises defined by a >=5% drawdown in the Representative Endowment (quarterly data). "
             "Lengths and time to recovery are expressed in integer quarters. "
             "Public Equity is included to compare drawdowns and recovery times. "
             "No individual endowments are shown here since they are annual data.")
    st.dataframe(pivot_df_q)

    latex_table_q = export_table_to_latex_pivot(
        pivot_df_q,
        description="Quarterly Crisis Periods (≥5% Drawdown for Representative Endowment)."
    )
    st.subheader("Latex Code for the Quarterly Crisis Table")
    st.code(latex_table_q, language="latex")

    # ====== SECOND TABLE: ANNUAL CRISES FOR NACUBO ======
    annual_cum_dict = {}

    if 'Average Endowment (NACUBO)' in endow_df.columns:
        nacubo_ret = endow_df['Average Endowment (NACUBO)'].dropna().sort_index()
        if not nacubo_ret.empty:
            # NACUBO cum
            nacubo_cum = (1 + nacubo_ret).cumprod()
            annual_cum_dict["Average Endowment (NACUBO)"] = nacubo_cum

            # We'll compare NACUBO crises to Public Equity, Representative Endowment, Yale, Stanford, Harvard
            annual_dates = nacubo_ret.index

            def reindex_annual(cum_series, new_index):
                if cum_series.empty:
                    return pd.Series(dtype=float)
                # forward-fill to match annual points
                return cum_series.reindex(new_index, method='pad').dropna()

            # Rep Endowment
            rep_annual = reindex_annual(rep_cum, annual_dates)
            annual_cum_dict["Representative Endowment"] = rep_annual

            # Public Equity
            if "Public Equity" in cum_dict_quarterly:
                pe_annual = reindex_annual(cum_dict_quarterly["Public Equity"], annual_dates)
                annual_cum_dict["Public Equity"] = pe_annual

            # Yale,Stanford,Harvard
            for e_col in ["Yale", "Stanford", "Harvard"]:
                if e_col in cum_dict_quarterly:
                    e_cum_ = cum_dict_quarterly[e_col]
                    e_annual = reindex_annual(e_cum_, annual_dates)
                    annual_cum_dict[e_col] = e_annual

            # Identify NACUBO crises
            nacubo_crises = find_crisis_periods_for_nacubo_annual(nacubo_ret, threshold=0.05)
            pivot_df_annual = pivot_crises_into_columns_annual(nacubo_crises, annual_cum_dict)

            st.subheader("Table 2: Annual Crisis Periods (≥5% Decline for NACUBO Average)")
            st.write("We define crises based on the Average Endowment (NACUBO) having a >=5% annual drawdown. "
                     "We compare how Representative Endowment and Public Equity (reindexed to annual), "
                     "as well as Yale, Stanford, and Harvard, performed in these periods. "
                     "Lengths and time to recovery are in integer years.")
            st.dataframe(pivot_df_annual)

            latex_table_annual = export_table_to_latex_pivot(
                pivot_df_annual,
                description="Annual Crisis Periods (≥5% Decline for NACUBO Average)."
            )
            st.subheader("Latex Code for the Annual Crisis Table")
            st.code(latex_table_annual, language="latex")

    # Build dd_dict for the quarterly chart
    dd_dict_q = {}
    for lbl, series_cum in cum_dict_quarterly.items():
        if series_cum.empty:
            dd_dict_q[lbl] = pd.Series(dtype=float)
        else:
            run_max = series_cum.cummax()
            dd = (series_cum - run_max) / run_max
            dd_dict_q[lbl] = dd

    st.subheader("Matplotlib Journal-Style Performance Chart (Quarterly Data)")
    fig_jof = create_performance_chart_jof_matplotlib(cum_dict_quarterly, dd_dict_q)
    st.pyplot(fig_jof)


if __name__ == '__main__':
    main()
