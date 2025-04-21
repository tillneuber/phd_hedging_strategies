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
    We add a 0.0% baseline row at 01‑Jul‑1999 for the quarterly data.
    """

    ########################################
    # Load the allocations
    ########################################
    alloc_cols = [
        'Year', 'Public Equity', 'PE/VC', 'Hedge Funds',
        'Real Assets & ILBs', 'Fixed Income', 'Private Credit', 'Cash'
    ]
    allocations = pd.read_csv('data/hist_endowment_saa.csv', sep=';', names=alloc_cols, header=0)

    # Convert wide→long
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
    returns_quarterly.index = returns_quarterly.index + pd.offsets.MonthEnd(0)

    # Convert string percentages to decimal
    returns_quarterly = returns_quarterly.apply(
        lambda col: col.map(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) else np.nan
        )
    )

    # ---- Add a zero‑return baseline row at 01‑Jul‑1999 ----
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
    Convert % → decimals, keep the date as end‑of‑period,
    and add a 0.0% baseline row at 01‑Jul‑1999 for the annual data as well.
    """
    df = pd.read_csv('data/individual_endowments.csv', sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Convert string percentages to decimal
    for col in df.columns:
        df[col] = df[col].apply(lambda x: float(str(x).replace('%', ''))/100 if pd.notnull(x) else np.nan)

    # ---- Add a zero‑return baseline row at 01‑Jul‑1999 ----
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
    For each date in date_index, find the 'Allocation' within (Start Date→End Date).
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
#  MATPLOTLIB JOIF‑STYLE PERFORMANCE CHART
########################################
def create_performance_chart_jof_matplotlib(cum_dict, dd_dict, rep_crises=None):
    """
    Two‑subplot figure: top=cumulative, bottom=drawdowns.
    Crisis intervals (from the Synthetic Endowment) are shaded in light‑grey.
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 4.6))

    # -- 1) Optionally shade crisis intervals -----------------------------
    if rep_crises:
        for crisis in rep_crises:
            start, end = crisis["Start"], crisis["End"]
            ax1.axvspan(start, end, color="lightgray", alpha=0.3, zorder=0)
            ax2.axvspan(start, end, color="lightgray", alpha=0.3, zorder=0)

    # -- 2) Helper to clean labels ----------------------------------------
    def rename_label(original_key):
        if original_key == "Synthetic Endowment Index":
            return "Synthetic Endowment Index"
        return original_key.replace("(Q)", "").replace("(Annual)", "").strip()

    # -- 3) Desired legend / plotting order -------------------------------
    desired_order = [
        "Synthetic Endowment Index",
        "Public Equity",
        "Average Endowment (NACUBO)",
        "Yale",
        "Stanford",
        "Harvard",
    ]

    # -- 4) Map original keys to cleaned labels ---------------------------
    rename_dict = {k: rename_label(k) for k in cum_dict.keys()}

    # -- 5) Build (label, key) pairs in the desired order -----------------
    sorted_labels = []
    for lab in desired_order:
        keys = [k for k, v in rename_dict.items() if v == lab]
        if keys:
            sorted_labels.append((lab, keys[0]))

    # -- 6) Choose line style & COLOR (now coloured for endowments) -------
    def choose_line_style(label):
        if label == "Synthetic Endowment Index":
            return dict(color="black", linestyle="solid", linewidth=1.3, zorder=3)
        if label == "Public Equity":
            return dict(color="dimgray", linestyle="solid", linewidth=1.3, zorder=3)
        if label == "Average Endowment (NACUBO)":
            return dict(color="darkorange", linestyle="dashdot", linewidth=1.3, zorder=2)
        if label == "Yale":
            return dict(color="navy", linestyle="dashed", linewidth=1.2, zorder=2)
        if label == "Stanford":
            return dict(color="seagreen", linestyle="dashed", linewidth=1.2, zorder=2)
        if label == "Harvard":
            return dict(color="firebrick", linestyle="dashed", linewidth=1.2, zorder=2)
        return dict(color="gray", linestyle="solid", linewidth=1.2, zorder=1)

    # ---------------- TOP: cumulative returns ---------------------------
    for lbl, key in sorted_labels:
        s = cum_dict[key]
        if s.empty:
            continue
        ax1.plot(s.index, s.values, label=lbl, **choose_line_style(lbl))

    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel("Cumulative Returns", fontsize=8)
    ax1.tick_params(labelsize=8)

    # ---------------- BOTTOM: drawdowns ---------------------------------
    for lbl, key in sorted_labels:
        dd = dd_dict[key]
        if dd.empty:
            continue
        ax2.plot(dd.index, dd.values, label=lbl, **choose_line_style(lbl))

    ax2.set_ylabel("Drawdowns", fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    # -- 7) X‑axis formatting --------------------------------------------
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # -- 8) Single shared legend (deduplicated) ---------------------------
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(pad=1.2)
    return fig



########################################
#  HELPER: Rounded Quarters/Years
########################################
def approximate_quarters_diff(ts_start, ts_end):
    """
    Approximate the difference in quarters by:
    1) Calculating the day difference,
    2) Converting to months using ~30.4375 days,
    3) Dividing by 3,
    4) Rounding to the nearest integer,
    5) Forcing a minimum of 1 if the result is 0.
    """
    days_diff = (ts_end - ts_start).days
    months = days_diff / 30.4375
    q_float = months / 3.0
    q_rounded = int(round(q_float))
    return max(1, q_rounded)

def approximate_years_diff(ts_start, ts_end):
    """
    Approximate the difference in years by:
    1) Calculating the day difference,
    2) Dividing by 365.25,
    3) Rounding to the nearest integer,
    4) Forcing a minimum of 1 if the result is 0.
    """
    days_diff = (ts_end - ts_start).days
    yrs_float = days_diff / 365.25
    yrs_rounded = int(round(yrs_float))
    return max(1, yrs_rounded)

########################################
#  CRISIS DETECTION (≥ 10 %) QUARTERLY
########################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.10):
    """
    Identify crisis periods in the Representative (Synthetic) Endowment's quarterly drawdown
    if ≥ threshold (10 %).

    For each crisis:
      'Start'  = last date before the drawdown begins (peak)
      'Trough' = date of maximum drawdown
      'End'    = date when the drawdown fully recovers to 0 %
    We skip crises where peak and trough coincide, or if the drawdown < threshold.
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
            # Set peak date as the previous date if dd was non‑negative
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
#    CRISIS DETECTION (≥ 10 %) ANNUAL NACUBO
########################################
def find_crisis_periods_for_nacubo_annual(returns_series, threshold=0.10):
    """
    Identify crisis periods in the Average Endowment (NACUBO) annual data if
    the peak‑to‑trough decline ≥ threshold (10 %).
    - Peak Date: The last date before the drawdown begins (local maximum).
    - Trough Date: The date when the maximum drawdown is reached.
    - Crises with identical peak and trough dates are skipped.
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


########################################
#   Time‑to‑Recovery Helpers
########################################
def find_time_to_recovery_q(cum_series, start_d, trough_d):
    """
    Returns an integer # of quarters from the trough until
    the series returns to or exceeds its value at 'start_d'.
    """
    if start_d not in cum_series.index or trough_d not in cum_series.index:
        return None
    peak_val = cum_series.loc[start_d]
    subset = cum_series.loc[trough_d:]
    rec_idx = subset[subset >= peak_val].index
    if len(rec_idx) == 0:
        return None
    return approximate_quarters_diff(trough_d, rec_idx[0])


def find_time_to_recovery_annual(cum_s, s_d, t_d):
    """
    Returns an integer # of years from the trough until
    the series returns to or exceeds its value at 's_d'.
    """
    if s_d not in cum_s.index or t_d not in cum_s.index:
        return None
    start_val = cum_s.loc[s_d]
    subset = cum_s.loc[t_d:]
    rec_idx = subset[subset >= start_val].index
    if len(rec_idx) == 0:
        return None
    return approximate_years_diff(t_d, rec_idx[0])


########################################
#  TABLE BUILDING – QUARTERLY
#  UPDATED to incorporate Recovery Date,
#  rename "Representative Endowment" → "Synthetic Endowment Index",
#  remove "q" suffix, etc.
########################################
def pivot_crises_into_columns(rep_crises, cum_dict):
    """
    Builds a pivoted crisis table based on a ≥ 10 % drawdown in the
    Synthetic Endowment Index.

    Includes:
     • Beginning Date
     • Trough Date
     • Recovery Date
     • Max Drawdown
     • Drawdown Length (in quarters, but no 'q' suffix)
     • Time to Recovery (in quarters, no suffix)
     • Public Equity Drawdown (Peak→Trough)
     • Public Equity Time to Recovery
    """
    if not rep_crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(rep_crises))]

    row_labels = [
        "Beginning Date",
        "Trough Date",
        "Recovery Date",
        "Max Drawdown",
        "Drawdown Length",
        "Time to Recovery",
        "---",
        "Public Equity Drawdown",
        "Public Equity Time to Recovery"
    ]
    data = {c: [""] * len(row_labels) for c in col_names}

    syn_key = "Synthetic Endowment Index"

    for i, crisis in enumerate(rep_crises):
        cname = col_names[i]
        start_dt = crisis['Start']
        trough_dt = crisis['Trough']
        end_dt = crisis['End']
        max_dd = crisis['Max Drawdown']

        # Row 0 → Beginning Date
        data[cname][0] = str(start_dt.date())
        # Row 1 → Trough Date
        data[cname][1] = str(trough_dt.date())
        # Row 2 → Recovery Date
        data[cname][2] = str(end_dt.date())
        # Row 3 → Max Drawdown
        data[cname][3] = f"{max_dd:.1%}"

        # Drawdown length
        q_len = approximate_quarters_diff(start_dt, trough_dt)
        data[cname][4] = f"{q_len}"

        # Time to Recovery
        if syn_key in cum_dict:
            syn_cum = cum_dict[syn_key]
            rep_ttr = find_time_to_recovery_q(syn_cum, start_dt, trough_dt)
            data[cname][5] = f"{rep_ttr}" if rep_ttr is not None else "n/a"
        else:
            data[cname][5] = "n/a"

        # Row 6 → (separator)
        data[cname][6] = "---"

        # Rows 7..8 → Public Equity
        if "Public Equity" in cum_dict:
            pe_cum = cum_dict["Public Equity"]
            if (start_dt in pe_cum.index) and (trough_dt in pe_cum.index):
                pk_val = pe_cum.loc[start_dt]
                th_val = pe_cum.loc[trough_dt]
                decline_e = (th_val - pk_val) / pk_val
                if decline_e > -1e-9:
                    data[cname][7] = "n/a"
                else:
                    data[cname][7] = f"{decline_e:.1%}"

                eq_ttr = find_time_to_recovery_q(pe_cum, start_dt, trough_dt)
                data[cname][8] = f"{eq_ttr}" if eq_ttr else "n/a"
            else:
                data[cname][7] = "n/a"
                data[cname][8] = "n/a"
        else:
            data[cname][7] = "n/a"
            data[cname][8] = "n/a"

    df_out = pd.DataFrame(data, index=row_labels)
    return df_out


########################################
#  TABLE BUILDING – ANNUAL NACUBO
########################################
def pivot_crises_into_columns_annual(nacubo_crises, annual_cum_dict):
    """
    For annual NACUBO‑based crises. Columns → Crisis1, Crisis2, ...
    Show Public Equity, Rep Endowment, Yale, Stanford, Harvard.
    Time to Recovery → integer # of years (rounded).
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
        "Synthetic Endowment Index Drawdown",
        "Synthetic Endowment Index Time to Recovery",
        "---3",
        "Yale Drawdown",
        "Yale Time to Recovery",
        "Stanford Drawdown",
        "Stanford Time to Recovery",
        "Harvard Drawdown",
        "Harvard Time to Recovery"
    ]
    data = {c: [""] * len(row_labels) for c in col_names}

    for i, crisis in enumerate(nacubo_crises):
        cname = col_names[i]
        start_dt = crisis['Start']
        trough_dt = crisis['Trough']
        end_dt = crisis['End']
        max_dd = crisis['Max Drawdown']

        data[cname][0] = str(start_dt.date())
        data[cname][1] = str(trough_dt.date())
        data[cname][2] = f"{max_dd:.1%}"

        # drawdown length
        yrs_len = approximate_years_diff(start_dt, trough_dt)
        data[cname][3] = f"{yrs_len} yrs"

        # time to recovery for NACUBO
        if "Average Endowment (NACUBO)" in annual_cum_dict:
            nacubo_cum = annual_cum_dict["Average Endowment (NACUBO)"]
            if (start_dt in nacubo_cum.index) and (trough_dt in nacubo_cum.index):
                ttr_n = find_time_to_recovery_annual(nacubo_cum, start_dt, trough_dt)
                data[cname][4] = f"{ttr_n} yrs" if ttr_n is not None else "n/a"
            else:
                data[cname][4] = "n/a"

        # fill for the row_map
        row_map = {
            "Public Equity": (6, 7),
            "Synthetic Endowment Index": (9, 10),
            "Yale": (12, 13),
            "Stanford": (14, 15),
            "Harvard": (16, 17)
        }
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
            data[cname][r_recovery] = f"{e_ttr_} yrs" if e_ttr_ is not None else "n/a"

    return pd.DataFrame(data, index=row_labels)


def export_table_to_latex_pivot(df, description=""):
    """
    Convert pivoted df to LaTeX code,
    using an alignment that doesn't break with & in the preamble.
    """
    columns = df.columns.tolist()
    row_labels = df.index.tolist()

    # first column → l, then all columns → c
    align_str = "l" + "".join(["c"] * len(columns))

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\" 
        r"\footnotesize{Crisis identification based on a 10\% drawdown threshold.}}"
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
    st.title("Approach #1: Two Separate Tables for Quarterly vs. Annual Crises")

    # 1) Load Data
    allocations, ret_q_full = load_and_preprocess_data()
    # Create a unified timeframe for allocations‑based calculations.
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q_full.copy())

    # NOTE: By default, pick a quarterly frequency that matches 'QE‑DEC'
    idx_q = pd.date_range(q_start, q_end, freq='QE-DEC')

    # Calculate Synthetic Endowment Index returns using the unified timeframe.
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)
    common_cols = hist_alloc_q.columns.intersection(ret_q.columns)
    hist_alloc_q, ret_q_hist = hist_alloc_q[common_cols], ret_q[common_cols]
    hist_endw_returns = (hist_alloc_q * ret_q_hist).sum(axis=1)
    rep_cum, rep_dd = calculate_cumulative_and_dd(hist_endw_returns)

    # Build the dictionary for the chart (quarterly).
    cum_dict_quarterly = {
        "Synthetic Endowment Index": rep_cum
    }

    # For Public Equity, attach the full returns so it starts in 1999 as well
    if 'Public Equity' in ret_q_full.columns:
        pe_returns = ret_q_full['Public Equity'].dropna()
        pe_cum, pe_dd = calculate_cumulative_and_dd(pe_returns)
        cum_dict_quarterly["Public Equity"] = pe_cum

    # Add the annual endowments too (Yale, Stanford, Harvard, NACUBO)
    endow_df = load_individual_endowments()
    for col in endow_df.columns:
        s_ = endow_df[col].dropna().sort_index()
        if not s_.empty:
            c_, d_ = calculate_cumulative_and_dd(s_)
            cum_dict_quarterly[col] = c_

    # ====== FIRST TABLE: QUARTERLY CRISES FOR SYNTHETIC ENDOWMENT INDEX ======
    rep_crises_q = find_crisis_periods_for_rep_endowment(rep_dd, threshold=0.10)
    pivot_df_q = pivot_crises_into_columns(rep_crises_q, cum_dict_quarterly)

    st.subheader("Table 1: Quarterly Crisis Periods (≥ 10 % Drawdown for Synthetic Endowment Index)")
    st.write(
        "This table shows crises defined by a ≥ 10 % drawdown in the Synthetic Endowment Index (quarterly data). "
        "We include a Recovery Date row, remove 'q' suffixes, and use the label 'Synthetic Endowment Index'. "
        "Public Equity is included to compare drawdowns and recovery times. "
        "No individual endowments are shown here because they are annual data."
    )
    st.dataframe(pivot_df_q)

    latex_table_q = export_table_to_latex_pivot(
        pivot_df_q,
        description="Quarterly Crisis Periods (≥ 10 % Drawdown for Synthetic Endowment Index)."
    )
    st.subheader("LaTeX Code for the Quarterly Crisis Table")
    st.code(latex_table_q, language="latex")

    # ====== SECOND TABLE: ANNUAL CRISES FOR NACUBO ======
    annual_cum_dict = {}

    if 'Average Endowment (NACUBO)' in endow_df.columns:
        nacubo_ret = endow_df['Average Endowment (NACUBO)'].dropna().sort_index()
        if not nacubo_ret.empty:
            # NACUBO cum
            nacubo_cum = (1 + nacubo_ret).cumprod()
            annual_cum_dict["Average Endowment (NACUBO)"] = nacubo_cum

            # Compare NACUBO crises to Public Equity, Synthetic Endowment Index, Yale, Stanford, Harvard
            annual_dates = nacubo_ret.index

            def reindex_annual(cum_series, new_index):
                if cum_series.empty:
                    return pd.Series(dtype=float)
                # forward‑fill to match annual points
                return cum_series.reindex(new_index, method='pad').dropna()

            # Synthetic Endowment Index
            rep_annual = reindex_annual(rep_cum, annual_dates)
            annual_cum_dict["Synthetic Endowment Index"] = rep_annual

            # Public Equity
            if "Public Equity" in cum_dict_quarterly:
                pe_annual = reindex_annual(cum_dict_quarterly["Public Equity"], annual_dates)
                annual_cum_dict["Public Equity"] = pe_annual

            # Yale, Stanford, Harvard
            for e_col in ["Yale", "Stanford", "Harvard"]:
                if e_col in cum_dict_quarterly:
                    e_cum_ = cum_dict_quarterly[e_col]
                    e_annual = reindex_annual(e_cum_, annual_dates)
                    annual_cum_dict[e_col] = e_annual

            # Identify NACUBO crises
            nacubo_crises = find_crisis_periods_for_nacubo_annual(nacubo_ret, threshold=0.10)
            pivot_df_annual = pivot_crises_into_columns_annual(nacubo_crises, annual_cum_dict)

            st.subheader("Table 2: Annual Crisis Periods (≥ 10 % Decline for NACUBO Average)")
            st.write(
                "Crises are defined by the Average Endowment (NACUBO) having a ≥ 10 % annual drawdown. "
                "We compare how the Synthetic Endowment Index and Public Equity (re‑indexed to annual), "
                "as well as Yale, Stanford and Harvard, performed in these periods. "
                "Lengths and time to recovery are expressed in integer years (rounded)."
            )
            st.dataframe(pivot_df_annual)

            latex_table_annual = export_table_to_latex_pivot(
                pivot_df_annual,
                description="Annual Crisis Periods (≥ 10 % Decline for NACUBO Average)."
            )
            st.subheader("LaTeX Code for the Annual Crisis Table")
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

    st.subheader("Matplotlib Journal‑Style Performance Chart (Quarterly Data)")
    # Pass the rep_crises_q so that we can shade those intervals
    fig_jof = create_performance_chart_jof_matplotlib(
        cum_dict_quarterly,
        dd_dict_q,
        rep_crises=rep_crises_q
    )
    st.pyplot(fig_jof)


if __name__ == '__main__':
    main()
