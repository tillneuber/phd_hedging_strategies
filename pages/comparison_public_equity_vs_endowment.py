"""
Streamlit app that:
  1) Loads your real CSV data for asset allocations + returns.
  2) Constructs a Synthetic Endowment Index (quarterly).
  3) Detects ≥X% crises (user-selectable threshold) separately for:
       a) Public Equity
       b) Synthetic Endowment Index
  4) Builds two crisis tables (one for each portfolio), each including:
       - Start date, Trough date, End date, Max Drawdown, Drawdown length, Time to recovery
       - Columns for each hedge strategy's peak->trough performance, plus rows to show
         both the average and median hedge performances across all crises for that portfolio.
  5) Plots separate cumulative + drawdown charts for each portfolio, shading its crisis periods.

Data Requirements:
    - data/hist_endowment_saa.csv (allocations)
    - data/quarterly_returns.csv (quarterly returns, must include a "Public Equity" column)
    - data/hedging_strategies.csv (hedge strategy returns)
All CSV columns with returns in percent must be converted to decimals.

Usage:
  streamlit run compare_public_equity_vs_endowment.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


###############################################################################
#                      HELPER: EXPORT TABLE TO LATEX
###############################################################################
def export_table_to_latex(df, description, threshold):
    """
    Convert a Pandas DataFrame (df) into a styled LaTeX table string.
    Includes:
    - \begin{table}, \centering, \begin{tiny}, ...
    - A caption with the 'description' and a note about the threshold.
    - A simple 'l' + 'c' * number_of_columns alignment approach.
    - \toprule, \midrule, \bottomrule for better table style.
    """

    # Gather columns and index as lists
    cols = df.columns.tolist()
    index_labels = df.index.tolist()

    # We'll do first column => l, then all columns => c
    # So if the DF has N columns, alignment is "l" + "c"*N
    align_str = "l" + "".join(["c" for _ in range(len(cols))])

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{description}}}\\"
        rf"\footnotesize{{Crisis identification based on a {threshold:.0f}\% drawdown threshold.}}}}"
    )
    lines.append(r"\label{table:" + description.replace(" ", "_").lower() + "}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + align_str + r"}")
    lines.append(r"\toprule")

    # Header line
    header_line = [""] + [str(c) for c in cols]
    lines.append(" & ".join(header_line) + r" \\")
    lines.append(r"\midrule")

    # Table body
    for idx_label in index_labels:
        row_vals = df.loc[idx_label, :]

        # If row_vals is a Series, we can iterate columns:
        row_cells = [str(idx_label)]  # first column: the row index
        for col in cols:
            val = row_vals[col]
            # Convert any percent sign to escaped
            if isinstance(val, str):
                val = val.replace("%", r"\%")
            row_cells.append(str(val))

        line = " & ".join(row_cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


###############################################################################
#                         LOADING MAIN DATA
###############################################################################

def load_endowment_allocations_and_returns():
    """
    Loads:
      1) 'hist_endowment_saa.csv' => TAA weights
         - columns: [Year, Public Equity, PE/VC, Hedge Funds, Real Assets & ILBs,
                     Fixed Income, Private Credit, Cash]
         - converted to long form with 'Asset Class' = X, 'Allocation' = Y
      2) 'quarterly_returns.csv' => each asset class's periodic returns
         - columns: Date;Public Equity;PE/VC;...
         - string percentages => decimals
    Also inserts a 0% row at 1999-07-01 if missing in returns.

    Returns:
      allocations_df (long form with columns
        ['Year','Asset Class','Allocation','Start Date','End Date'])
      returns_q (pd.DataFrame, index=Date, columns=asset classes in decimal returns)
    """

    # ============== 1) LOAD ALLOCATIONS ==============
    alloc_cols = [
        "Year", "Public Equity", "PE/VC", "Hedge Funds",
        "Real Assets & ILBs", "Fixed Income", "Private Credit", "Cash"
    ]
    alloc = pd.read_csv("data/hist_endowment_saa.csv", sep=";", names=alloc_cols, header=0)

    # Convert wide->long
    alloc_long = alloc.melt(
        id_vars=["Year"], var_name="Asset Class", value_name="Allocation"
    )
    # Convert to numeric
    alloc_long["Allocation"] = pd.to_numeric(alloc_long["Allocation"], errors="coerce")

    # For each Year, Start Date = that year + 6 months (June 30),
    # End Date = +1 year - 1 day
    alloc_long["Start Date"] = pd.to_datetime(alloc_long["Year"].astype(str)) + pd.DateOffset(months=6)
    alloc_long["End Date"] = alloc_long["Start Date"] + pd.DateOffset(years=1) - pd.DateOffset(days=1)

    # ============== 2) LOAD QUARTERLY RETURNS ==============
    ret_q = pd.read_csv("data/quarterly_returns.csv", sep=";", header=0)
    ret_q["Date"] = pd.to_datetime(ret_q["Date"], format="%d.%m.%Y", errors="coerce")
    ret_q.set_index("Date", inplace=True)

    # Shift index to month-end
    ret_q.index = ret_q.index + pd.offsets.MonthEnd(0)

    # Convert string percentages => decimals
    for col in ret_q.columns:
        ret_q[col] = ret_q[col].apply(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) else np.nan
        )

    # Insert 0% row at 1999-07-01 if needed
    base_dt = pd.Timestamp("1999-07-01")
    if base_dt not in ret_q.index:
        zero_vals = {c: 0.0 for c in ret_q.columns}
        ret_q.loc[base_dt] = zero_vals
    ret_q.sort_index(inplace=True)

    return alloc_long, ret_q


def unify_timeframe(allocations, returns_q):
    """
    Align time range by comparing allocations' earliest start and latest end
    to returns' earliest and latest index.

    Returns:
      filtered_allocations, filtered_returns, start_date, end_date
    """
    earliest_alloc_start = allocations["Start Date"].min()
    latest_alloc_end     = allocations["End Date"].max()
    earliest_ret         = returns_q.index.min()
    latest_ret           = returns_q.index.max()

    overall_start = max(earliest_alloc_start, earliest_ret)
    overall_end   = min(latest_alloc_end, latest_ret)

    # Filter allocations
    a_mask = (allocations["End Date"] >= overall_start) & (allocations["Start Date"] <= overall_end)
    allocations = allocations[a_mask]

    # Filter returns
    r_mask = (returns_q.index >= overall_start) & (returns_q.index <= overall_end)
    returns_q = returns_q[r_mask]

    return allocations, returns_q, overall_start, overall_end


def map_allocations_to_periods(allocations_df, date_index):
    """
    For each date in date_index, find the relevant allocation if date is between
    (Start Date, End Date) for that row. We'll produce a wide df: one row per date,
    col per asset, value=allocation.
    """
    out = pd.DataFrame(index=date_index)
    assets = allocations_df["Asset Class"].unique()
    for asset in assets:
        asset_sub = allocations_df[allocations_df["Asset Class"] == asset]
        for _, row in asset_sub.iterrows():
            date_rng = pd.date_range(row["Start Date"], row["End Date"], freq=date_index.freq)
            out.loc[out.index.intersection(date_rng), asset] = row["Allocation"]
    return out.fillna(0.0)


def calculate_cumulative_and_dd(return_series):
    """
    Given a decimal return series (periodic),
    build a cumulative index, and drawdown series.
    """
    cum = (1 + return_series).cumprod()
    run_max = cum.cummax()
    dd = (cum - run_max) / run_max
    return cum, dd


###############################################################################
#                    LOAD HEDGING STRATEGIES
###############################################################################
def load_hedging_strategies():
    """
    Loads 'data/hedging_strategies.csv' with columns:
      Date;Global Macro;Hedge Funds;Tail Risk HF;Time Series Momentum (Very Fast);...
    Convert string % => decimal, parse Date => monthly or quarterly.

    We'll assume it's at least monthly; we then resample to Q in the main app.
    """
    file_path = "data/hedging_strategies.csv"
    df = pd.read_csv(file_path, sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)

    # Move index to end of month
    df.index = df.index + pd.offsets.MonthEnd(0)

    # Convert strings => decimals
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) and x != '' else np.nan
        )

    df.sort_index(inplace=True)
    return df


###############################################################################
#                   CRISIS DETECTION (EXAMPLE LOGIC)
###############################################################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.05):
    """
    Identify crisis periods in the drawdown if ≥ threshold (default 5% here).
    This logic ensures:
      - A crisis begins when dd first goes below 0.
      - 'Start' = the last date dd >= 0 prior to that (or same date if not found).
      - 'Trough' = minimum dd within that crisis.
      - Crisis ends when dd returns to 0.
      - Only record if the max drawdown <= -threshold and start != trough.
      - If the crisis doesn't fully recover by the final date, we consider that
        an 'End' if max_dd <= -threshold.
    """
    ds = dd_series.dropna().copy()
    in_crisis = False
    crises = []
    start_date = None
    trough_date = None
    max_dd = 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            # Enter crisis
            in_crisis = True
            idx = ds.index.get_loc(date)
            # The "peak" is the last date dd >= 0 prior to dropping
            if idx > 0 and ds.iloc[idx - 1] >= 0:
                start_date = ds.index[idx - 1]
            else:
                start_date = date
            trough_date = date
            max_dd = dd_val
        elif in_crisis:
            # Update trough
            if dd_val < max_dd:
                max_dd = dd_val
                trough_date = date

            # Crisis ends if dd_val == 0
            if dd_val == 0:
                in_crisis = False
                end_date = date
                # Check threshold & distinct peak/trough
                if (max_dd <= -threshold) and (start_date != trough_date):
                    crises.append({
                        "Start": start_date,
                        "Trough": trough_date,
                        "End": end_date,
                        "Max Drawdown": max_dd
                    })
                start_date, trough_date, max_dd = None, None, 0.0

    # If still in crisis at the end of the series
    if in_crisis:
        end_date = ds.index[-1]
        if (max_dd <= -threshold) and (start_date != trough_date):
            crises.append({
                "Start": start_date,
                "Trough": trough_date,
                "End": end_date,
                "Max Drawdown": max_dd
            })

    return crises


###############################################################################
#   BUILD TABLE: CRISIS START/TROUGH/END + HEDGE PERF (PEAK->TROUGH)
###############################################################################
def peak_trough_return(cum_series, start_dt, trough_dt):
    """
    If start_dt and trough_dt are in cum_series, return
    (cum[trough_dt]/cum[start_dt] - 1). Otherwise NaN.
    """
    if (start_dt not in cum_series.index) or (trough_dt not in cum_series.index):
        return np.nan
    start_val = cum_series.loc[start_dt]
    trough_val = cum_series.loc[trough_dt]
    if start_val == 0:
        return np.nan
    return (trough_val / start_val) - 1.0


def approximate_quarters_diff(start_dt, end_dt):
    """
    Approx #quarters between two datetimes, rounding to nearest int,
    min 1 if there's any difference.
    """
    days_ = (end_dt - start_dt).days
    if days_ < 0:
        return 0
    months_ = days_ / 30.4375
    q_ = months_ / 3
    q_int = int(round(q_))
    return max(q_int, 1)


def build_crisis_table_with_hedges(crises_list, portfolio_name, cum_hedge_dict):
    """
    For each crisis, create a row with:
      [Portfolio, Crisis #, Start, Trough, End, MaxDrawdown, 
       DrawdownLengthQ, TimeToRecoveryQ, HEDGE: Strategy1, ..., HEDGE: StrategyN]

    Then add two "ALL" rows:
      - One for average (ALL - AVERAGE)
      - One for median (ALL - MEDIAN)
    across all crises for that portfolio, for each hedge strategy.
    """
    hedge_strats = list(cum_hedge_dict.keys())
    rows = []

    # 1) Build one row per crisis
    for i, cr in enumerate(crises_list, start=1):
        s_dt = cr["Start"]
        t_dt = cr["Trough"]
        e_dt = cr["End"]
        mdd = cr["Max Drawdown"]

        dd_len_q = approximate_quarters_diff(s_dt, t_dt)
        rec_len_q = approximate_quarters_diff(t_dt, e_dt)

        row_data = {
            "Portfolio": portfolio_name,
            "Crisis #": i,
            "Start": str(s_dt.date()),
            "Trough": str(t_dt.date()),
            "End": str(e_dt.date()),
            "Max Drawdown": f"{mdd:.1%}",
            "Drawdown Length (Qtrs)": dd_len_q,
            "Time to Recovery (Qtrs)": rec_len_q
        }

        # Hedge columns => peak->trough
        for hn in hedge_strats:
            pt_ret = peak_trough_return(cum_hedge_dict[hn], s_dt, t_dt)
            if pd.isna(pt_ret):
                row_data[f"HEDGE: {hn}"] = "n/a"
            else:
                row_data[f"HEDGE: {hn}"] = f"{pt_ret:.1%}"

        rows.append(row_data)

    df_crises = pd.DataFrame(rows)
    if df_crises.empty:
        return df_crises

    # 2) Add rows for average & median across crises
    hedge_cols = [c for c in df_crises.columns if c.startswith("HEDGE: ")]

    # Prepare empty "ALL - AVERAGE" / "ALL - MEDIAN" row placeholders
    avg_row = {
        "Portfolio": portfolio_name,
        "Crisis #": "ALL - AVERAGE",
        "Start": "",
        "Trough": "",
        "End": "",
        "Max Drawdown": "--AVG--",
        "Drawdown Length (Qtrs)": "",
        "Time to Recovery (Qtrs)": ""
    }
    med_row = {
        "Portfolio": portfolio_name,
        "Crisis #": "ALL - MEDIAN",
        "Start": "",
        "Trough": "",
        "End": "",
        "Max Drawdown": "--MED--",
        "Drawdown Length (Qtrs)": "",
        "Time to Recovery (Qtrs)": ""
    }

    # For each hedge column, compute average & median
    for hc in hedge_cols:
        numeric_vals = []
        for val_str in df_crises[hc]:
            # If it's a string ending with '%', parse to float
            if isinstance(val_str, str) and val_str.endswith('%'):
                try:
                    f_ = float(val_str.replace('%', '')) / 100
                    numeric_vals.append(f_)
                except:
                    pass
        # If no numeric values found, assign 'n/a'
        if not numeric_vals:
            avg_row[hc] = "n/a"
            med_row[hc] = "n/a"
        else:
            avg_ = np.mean(numeric_vals)
            med_ = np.median(numeric_vals)
            avg_row[hc] = f"{avg_:.1%}"
            med_row[hc] = f"{med_:.1%}"

    # Add both "ALL - AVERAGE" and "ALL - MEDIAN" rows to df_crises
    df_crises = pd.concat([df_crises, pd.DataFrame([avg_row, med_row])], ignore_index=True)
    return df_crises


###############################################################################
#                    PLOTTING
###############################################################################
def plot_portfolio_with_crises(cum_series, dd_series, crises_list, label="Portfolio"):
    """
    2-subplot chart: top => cumulative, bottom => drawdown,
    shading the crises intervals for that portfolio only.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 5), sharex=True)

    # Shade crisis intervals
    for cr in crises_list:
        ax1.axvspan(cr["Start"], cr["End"], color='lightgray', alpha=0.3)
        ax2.axvspan(cr["Start"], cr["End"], color='lightgray', alpha=0.3)

    # Plot cumulative
    ax1.plot(cum_series.index, cum_series.values, label=label, color='black', lw=1.3)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()

    # Plot drawdown
    ax2.plot(dd_series.index, dd_series.values, label=f"{label} Drawdown", color='gray', lw=1.2)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("Drawdown")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(label)
    fig.tight_layout(pad=1.2)
    return fig


###############################################################################
#                         MAIN STREAMLIT APP
###############################################################################
def main():
    st.title("Comparison: Public Equity vs. Synthetic Endowment Crises + Hedge Performance")

    # Let the user pick the crisis threshold with a slider
    st.sidebar.write("### Crisis Detection Threshold")
    threshold_percentage = st.sidebar.slider(
        "Select the crisis threshold (%)",
        min_value=1,
        max_value=25,
        value=10,  # <-- DEFAULT IS NOW 10%
        step=1
    )
    crisis_threshold = threshold_percentage / 100.0

    # 1) Load main data for Synthetic Endowment
    st.write("**Step 1:** Load allocations + returns, unify timeframe, build Synthetic Endowment Index.")
    allocations_long, ret_q_full = load_endowment_allocations_and_returns()
    allocations_long, ret_q, q_start, q_end = unify_timeframe(allocations_long, ret_q_full.copy())

    # Build a quarterly date range
    idx_q = pd.date_range(q_start, q_end, freq='Q')
    mapped_alloc_q = map_allocations_to_periods(allocations_long, idx_q)

    # Only keep columns present in both
    valid_assets = mapped_alloc_q.columns.intersection(ret_q.columns)
    mapped_alloc_q, ret_q = mapped_alloc_q[valid_assets], ret_q[valid_assets]

    # Compute Synthetic Endowment returns
    endw_returns = (mapped_alloc_q * ret_q).sum(axis=1).dropna()
    cum_endw, dd_endw = calculate_cumulative_and_dd(endw_returns)

    # Double-check that "Public Equity" is actually within the (now) truncated data
    if "Public Equity" not in ret_q.columns:
        st.error("No 'Public Equity' found in the unified timeframe. Exiting.")
        return

    # 2) Build the Public Equity series from the **same** unified timeframe
    st.write("**Step 2:** Identify Public Equity returns from the unified timeframe.")
    pe_series = ret_q["Public Equity"].dropna()
    cum_pe, dd_pe = calculate_cumulative_and_dd(pe_series)

    # 3) Load hedging strategies, unify timeframe, resample if needed
    st.write("**Step 3:** Load hedging strategies, resample or slice to unify with the final timeframe.")
    hedge_df = load_hedging_strategies()
    # Slice to overall range
    mask_hedge = (hedge_df.index >= q_start) & (hedge_df.index <= q_end)
    hedge_df = hedge_df[mask_hedge]

    # If monthly data, we resample to Q
    hedge_q = hedge_df.resample('Q').last().dropna(how='all')

    # Build cumulative indexes for each hedge strategy
    cum_hedges = {}
    for c_ in hedge_q.columns:
        s_ = hedge_q[c_].dropna()
        if len(s_) > 1:
            ch, _ = calculate_cumulative_and_dd(s_)
            cum_hedges[c_] = ch

    # 4) Detect crises for each portfolio using the selected threshold
    st.write(f"**Step 4:** Detect crises (≥{threshold_percentage}% drawdown) for Public Equity and Synthetic Endowment.")
    pe_crises = find_crisis_periods_for_rep_endowment(dd_pe, threshold=crisis_threshold)
    endw_crises = find_crisis_periods_for_rep_endowment(dd_endw, threshold=crisis_threshold)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Public Equity Crises**")
        if pe_crises:
            st.write(pe_crises)
        else:
            st.write("No crises found for Public Equity at this threshold.")

    with col_b:
        st.markdown("**Endowment Crises**")
        if endw_crises:
            st.write(endw_crises)
        else:
            st.write("No crises found for the Synthetic Endowment at this threshold.")

    # 5) Build Hedge Perf Tables (with both average & median)
    st.subheader("Public Equity Crisis Table + Hedge Peak->Trough Performance")
    if pe_crises:
        df_pe_crisis = build_crisis_table_with_hedges(pe_crises, "Public Equity", cum_hedges)
        st.dataframe(df_pe_crisis)

        # Generate and display LaTeX code for the Public Equity crisis table
        latex_pe = export_table_to_latex(df_pe_crisis,
                                         "Public Equity Crisis Table",
                                         threshold_percentage)
        st.markdown("**LaTeX Code for Public Equity Crisis Table**:")
        st.code(latex_pe, language="latex")

    else:
        st.write("No Public Equity crises => no table.")

    st.subheader("Endowment Crisis Table + Hedge Peak->Trough Performance")
    if endw_crises:
        df_endw_crisis = build_crisis_table_with_hedges(endw_crises, "Synthetic Endowment", cum_hedges)
        st.dataframe(df_endw_crisis)

        # Generate and display LaTeX code for the Endowment crisis table
        latex_endw = export_table_to_latex(df_endw_crisis,
                                           "Endowment Crisis Table",
                                           threshold_percentage)
        st.markdown("**LaTeX Code for Endowment Crisis Table**:")
        st.code(latex_endw, language="latex")

    else:
        st.write("No Endowment crises => no table.")

    st.markdown(
        """
**Interpretation**:
- For each crisis, we show:
  - Start, Trough, End dates
  - Max Drawdown (peak-to-trough)
  - Drawdown Length = quarters from Start -> Trough
  - Time to Recovery = quarters from Trough -> End
  - For each hedge strategy, "HEDGE: X" column shows that strategy's peak->trough performance.
- The final rows **"ALL - AVERAGE"** and **"ALL - MEDIAN"** summarize hedge performance across all crises.
"""
    )

    # 6) Plot each portfolio + shading
    st.subheader("Cumulative + Drawdown Charts with Crisis Shading")

    # Public Equity
    fig_pe = plot_portfolio_with_crises(cum_pe, dd_pe, pe_crises, label="Public Equity")
    st.pyplot(fig_pe)

    # Synthetic Endowment
    fig_endw = plot_portfolio_with_crises(cum_endw, dd_endw, endw_crises, label="Synthetic Endowment")
    st.pyplot(fig_endw)

    st.info(
        """
By using the same unified timeframe and the same crisis-detection logic for both 
Public Equity and the Synthetic Endowment Index, the table and the chart should 
now be consistent. 
You can see how each portfolio's crises differ in timing and severity, and how 
various hedge strategies performed from peak-to-trough of each crisis.
"""
    )


if __name__ == "__main__":
    main()
