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
    # Drop columns fully NaN
    df = df.dropna(axis=1, how='all')
    if df.empty:
        return df

    # Find earliest/largest start date and latest/smallest end date
    starts = []
    ends = []
    for col in df.columns:
        col_series = df[col].dropna()
        if not col_series.empty:
            starts.append(col_series.index.min())
            ends.append(col_series.index.max())

    if not starts or not ends:
        # No valid columns
        return pd.DataFrame()

    max_start = max(starts)
    min_end = min(ends)

    if max_start > min_end:
        # Negative overlap => empty
        return pd.DataFrame()

    df_sub = df.loc[max_start:min_end].copy()
    if df_sub.empty:
        return df_sub

    # Drop columns that remain fully NaN in the subrange
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
            mean_ = series.mean() * 100
            std_ = series.std() * 100
            min_ = series.min() * 100
            max_ = series.max() * 100
            skew_ = stats.skew(series, bias=False)
            kurt_ = stats.kurtosis(series, bias=False)

            results[col] = {
                'Mean (%)': mean_,
                'StdDev (%)': std_,
                'Min (%)': min_,
                'Max (%)': max_,
                'Skewness': skew_,
                'Excess Kurtosis': kurt_
            }

    return pd.DataFrame(results).T

def latex_escape_percent(x: float) -> str:
    """Format float with 2 decimals and append '\\%' for LaTeX."""
    return f"{x:.2f}\\%"

def convert_df_to_latex(df: pd.DataFrame,
                        date_start,
                        date_end,
                        frequency_str: str,
                        caption: str = "Descriptive Statistics",
                        label: str = "table:desc_stats",
                        note: str = "Mean, Std Dev, Min, Max in \\%, Skewness, Kurtosis unitless."
                       ) -> str:
    """
    Convert the descriptive-stats DataFrame into LaTeX code 
    suitable for finance journals, escaping '%' properly.
    """
    if df.empty or date_start is None or date_end is None:
        return "No data to generate LaTeX table."

    time_period_str = f"Period: {date_start.date()} to {date_end.date()} ({frequency_str} data). "
    note = note.replace('%', '\\%')  # ensure any '%' in note is escaped

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tiny}")
    lines.append(
        rf"\caption{{\normalsize{{{caption}}}\\ \footnotesize{{{time_period_str}{note}}}}}"
    )
    lines.append(rf"\label{{{label}}}")
    cols = len(df.columns)
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l" + "c"*cols + r"}")
    lines.append(r"\toprule")

    # Table header
    header_cols = [""] + list(df.columns)
    header_escaped = [h.replace('%', '\\%') for h in header_cols]
    lines.append(" & ".join(header_escaped) + r" \\")
    lines.append(r"\midrule")

    # Table rows
    for idx, row in df.iterrows():
        row_name = str(idx)
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
        line = row_name + " & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{tiny}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

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
        # Display stats
        q_start, q_end = q_shared.index.min(), q_shared.index.max()
        desc_stats_q = compute_descriptive_stats(q_shared)
        st.dataframe(desc_stats_q.style.format("{:.2f}"))

        # Latex code
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

    # 4) Annual endowment overlap
    st.subheader("Annual Endowments: Strict Overlap")
    a_shared = get_longest_shared_period_strict(endow_df)
    if a_shared.empty:
        st.warning("No overlapping date range found for the annual endowment dataset.")
    else:
        a_start, a_end = a_shared.index.min(), a_shared.index.max()
        desc_stats_a = compute_descriptive_stats(a_shared)
        st.dataframe(desc_stats_a.style.format("{:.2f}"))

        # Latex code
        latex_a = convert_df_to_latex(
            df=desc_stats_a,
            date_start=a_start,
            date_end=a_end,
            frequency_str="Annual",
            caption="Annual Endowment Descriptive Stats (Strict Overlap)",
            label="table:annual_strict_stats"
        )
        st.write("**LaTeX Code (Annual)**:")
        st.code(latex_a, language="latex")

if __name__ == "__main__":
    main()
