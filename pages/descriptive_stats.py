import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# Import your main app's data-loading functions
from streamlit_app import load_and_preprocess_data, load_individual_endowments

###############################################################################
#                         TIME-PERIOD INTERSECTION
###############################################################################

def get_longest_shared_period_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict intersection approach:
      1) Identify each column's earliest and latest valid dates.
      2) Overlap = [max_of_starts : min_of_ends].
      3) Slice df to that range.
      4) Drop columns with no valid data in that slice.
    """
    df = df.dropna(axis=1, how='all')
    if df.empty:
        return df

    starts, ends = [], []
    for col in df.columns:
        col_series = df[col].dropna()
        if not col_series.empty:
            starts.append(col_series.index.min())
            ends.append(col_series.index.max())
    if not starts or not ends:
        return pd.DataFrame()
    max_start, min_end = max(starts), min(ends)
    if max_start > min_end:
        return pd.DataFrame()
    df_sub = df.loc[max_start:min_end].copy()
    if df_sub.empty:
        return df_sub
    columns_to_drop = [col for col in df_sub.columns if df_sub[col].dropna().empty]
    df_sub.drop(columns=columns_to_drop, inplace=True)
    return df_sub

###############################################################################
#                  DESCRIPTIVE STATS + LATEX EXPORT
###############################################################################

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column:
      - Mean
      - StdDev
      - Min
      - Max
      - Skewness
      - Excess Kurtosis
    Values in decimal form are multiplied by 100 => percentages.
    """
    df = df.dropna(axis=1, how='all')
    results = {}
    for col in df.columns:
        series = df[col].dropna()
        if not series.empty:
            results[col] = {
                'Mean (%)': series.mean() * 100,
                'StdDev (%)': series.std() * 100,
                'Min (%)': series.min() * 100,
                'Max (%)': series.max() * 100,
                'Skewness': stats.skew(series, bias=False),
                'Excess Kurtosis': stats.kurtosis(series, bias=False)
            }
    return pd.DataFrame(results).T

def latex_escape_percent(x: float) -> str:
    return f"{x:.2f}\\%"

def convert_df_to_latex(df: pd.DataFrame,
                        date_start,
                        date_end,
                        frequency_str: str,
                        caption: str = "Descriptive Statistics",
                        label: str = "table:desc_stats",
                        note: str = "Mean, Std Dev, Min, Max in \\%, Skewness, Kurtosis unitless."
                       ) -> str:
    if df.empty or date_start is None or date_end is None:
        return "No data to generate LaTeX table."
    time_period_str = f"Period: {date_start.date()} to {date_end.date()} ({frequency_str} data). "
    note = note.replace('%', '\\%')
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\begin{tiny}",
        rf"\caption{{\normalsize{{{caption}}}\\ \footnotesize{{{time_period_str}{note}}}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l" + "c" * len(df.columns) + r"}",
        r"\toprule"
    ]
    header_cols = [""] + list(df.columns)
    header_escaped = [h.replace('%', '\\%') for h in header_cols]
    lines.append(" & ".join(header_escaped) + r" \\")
    lines.append(r"\midrule")
    for idx, row in df.iterrows():
        row_vals = []
        for col in df.columns:
            val = row[col]
            if pd.notnull(val):
                if "(%)" in col:
                    row_vals.append(latex_escape_percent(val))
                else:
                    row_vals.append(f"{val:.2f}")
            else:
                row_vals.append("")
        lines.append(str(idx) + " & " + " & ".join(row_vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{tiny}", r"\end{table}"])
    return "\n".join(lines)

###############################################################################
#            HEDGING STRATEGIES DATA LOADING FOR STATS
###############################################################################

def load_hedging_strategies_for_stats() -> pd.DataFrame:
    file_path = "data/hedging_strategies.csv"
    df = pd.read_csv(file_path, sep=';', header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)
    df.index = df.index + pd.offsets.MonthEnd(0)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: float(str(x).replace('%', ''))/100 if pd.notnull(x) and x != '' else np.nan)
    df.sort_index(inplace=True)
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
#                              STREAMLIT PAGE
###############################################################################

def main():
    st.title("Descriptive Statistics (Ascending Index, Overlap, and LaTeX)")
    
    # 1) Load data
    allocations, ret_q = load_and_preprocess_data()
    endow_df = load_individual_endowments()
    
    # 2) Ensure ascending index
    ret_q.sort_index(inplace=True)
    endow_df.sort_index(inplace=True)
    
    # 3) Quarterly overlap
    st.subheader("Quarterly Returns: Strict Overlap")
    q_shared = get_longest_shared_period_strict(ret_q)
    if q_shared.empty:
        st.warning("No overlapping date range found for the quarterly dataset.")
    else:
        q_start, q_end = q_shared.index.min(), q_shared.index.max()
        desc_stats_q = compute_descriptive_stats(q_shared)
        st.dataframe(desc_stats_q.style.format("{:.2f}"))
        latex_q = convert_df_to_latex(
            df=desc_stats_q,
            date_start=q_start,
            date_end=q_end,
            frequency_str="Quarterly",
            caption="Quarterly Returns Descriptive Stats (Strict Overlap)",
            label="table:quarterly_strict_stats"
        )
        st.write("**LaTeX Code (Quarterly)**:")
        st.code(latex_q, language="latex")
    
    # 4) Annual endowment overlap (including Representative Endowment)
    st.subheader("Annual Endowments: Strict Overlap")
    a_shared = get_longest_shared_period_strict(endow_df)
    
    # --- Compute Representative Endowment Annual Returns ---
    earliest_alloc_start = allocations['Start Date'].min()
    latest_alloc_end = allocations['End Date'].max()
    earliest_ret_q = ret_q.index.min()
    latest_ret_q = ret_q.index.max()
    rep_start = max(earliest_alloc_start, earliest_ret_q)
    rep_end = min(latest_alloc_end, latest_ret_q)
    allocations_filtered = allocations[(allocations['Start Date'] <= rep_end) & (allocations['End Date'] >= rep_start)]
    ret_q_filtered = ret_q[rep_start:rep_end]
    # Use a quarterly date range for representative endowment calculation
    idx_q = pd.date_range(rep_start, rep_end, freq='Q-DEC')
    # Mapping function without converting allocation values (use them as provided)
    def map_allocations_to_periods(allocations_df, date_index):
        allocations_mapped = pd.DataFrame(index=date_index)
        for asset in allocations_df['Asset Class'].unique():
            asset_df = allocations_df[allocations_df['Asset Class'] == asset]
            for _, row in asset_df.iterrows():
                period = pd.date_range(start=row['Start Date'], end=row['End Date'], freq=date_index.freq)
                allocations_mapped.loc[allocations_mapped.index.isin(period), asset] = row['Allocation']
        return allocations_mapped.fillna(0)
    hist_alloc_q = map_allocations_to_periods(allocations_filtered, idx_q)
    common_cols = hist_alloc_q.columns.intersection(ret_q_filtered.columns)
    hist_alloc_q = hist_alloc_q[common_cols]
    ret_q_hist = ret_q_filtered[common_cols]
    # Compute weighted quarterly returns for the representative endowment
    hist_endw_returns = (hist_alloc_q * ret_q_hist).sum(axis=1)
    # Compute annual return by compounding quarterly returns:
    rep_annual_return = (1 + hist_endw_returns).resample('A-JUN').prod() - 1
    rep_annual_df = rep_annual_return.to_frame(name="Representative Endowment")
    
    # --- Merge individual and representative annual data ---
    combined_df = pd.concat([a_shared, rep_annual_df], axis=1, join='inner')
    
    if combined_df.empty:
        st.warning("No overlapping date range found for the combined annual endowment dataset.")
    else:
        a_start, a_end = combined_df.index.min(), combined_df.index.max()
        desc_stats_a = compute_descriptive_stats(combined_df)
        st.dataframe(desc_stats_a.style.format("{:.2f}"))
        latex_a = convert_df_to_latex(
            df=desc_stats_a,
            date_start=a_start,
            date_end=a_end,
            frequency_str="Annual",
            caption="Annual Endowment Descriptive Stats (Strict Overlap) including Representative Endowment",
            label="table:annual_strict_stats"
        )
        st.write("**LaTeX Code (Annual Endowments including Representative Endowment)**:")
        st.code(latex_a, language="latex")
    
    # 5) Hedging Strategies overlap
    st.subheader("Hedging Strategies: Strict Overlap")
    hs_df = load_hedging_strategies_for_stats()
    hs_shared = get_longest_shared_period_strict(hs_df)
    if hs_shared.empty:
        st.warning("No overlapping date range found for the hedging strategies dataset.")
    else:
        hs_start, hs_end = hs_shared.index.min(), hs_shared.index.max()
        desc_stats_hs = compute_descriptive_stats(hs_shared)
        st.dataframe(desc_stats_hs.style.format("{:.2f}"))
        latex_hs = convert_df_to_latex(
            df=desc_stats_hs,
            date_start=hs_start,
            date_end=hs_end,
            frequency_str="Monthly",
            caption="Hedging Strategies Descriptive Stats (Strict Overlap)",
            label="table:hedging_strict_stats"
        )
        st.write("**LaTeX Code (Hedging Strategies)**:")
        st.code(latex_hs, language="latex")

if __name__ == "__main__":
    main()
