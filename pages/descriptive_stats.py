###############################################################################
#                                IMPORTS
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# ─── project‑specific helpers ────────────────────────────────────────────────
from streamlit_app import (
    load_and_preprocess_data,     # returns (allocations, quarterly_returns)
    load_individual_endowments,   # returns annual endowment dataframe
)

###############################################################################
#                     TIME‑PERIOD INTERSECTION (STRICT)
###############################################################################
def get_longest_shared_period_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Return the slice where *all* series overlap (non‑NaN)."""
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return df

    starts, ends = [], []
    for col in df.columns:
        s = df[col].dropna()
        if not s.empty:
            starts.append(s.index.min())
            ends.append(s.index.max())

    if not starts or not ends:
        return pd.DataFrame()

    max_start, min_end = max(starts), min(ends)
    if max_start > min_end:
        return pd.DataFrame()

    df_sub = df.loc[max_start:min_end].copy()
    df_sub.dropna(axis=1, how="all", inplace=True)
    return df_sub


###############################################################################
#                DESCRIPTIVE STATS (+ annualisation)
###############################################################################
def compute_descriptive_stats(
    df: pd.DataFrame,
    ann_factor: int = 1,                 # 4 for quarterly, 12 for monthly …
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      • Annualised Mean %, Std‑Dev %
      • Skewness, Excess Kurtosis
    """
    df = df.dropna(axis=1, how="all")
    out = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        mean_ann = (1 + s.mean()) ** ann_factor - 1
        std_ann  = s.std() * np.sqrt(ann_factor)
        out[col] = {
            "Mean (%)": mean_ann * 100,
            "StdDev (%)": std_ann * 100,
            "Skewness": stats.skew(s, bias=False),
            "Excess Kurtosis": stats.kurtosis(s, bias=False),
        }
    return pd.DataFrame(out).T


###############################################################################
#                         LaTeX HELPERS
###############################################################################
def latex_escape_percent(x: float) -> str:
    return f"{x:.2f}\\%"

def df_to_latex_rows(df: pd.DataFrame) -> list[str]:
    rows = []
    for idx, row in df.iterrows():
        vals = [
            latex_escape_percent(v) if "(%)" in col else f"{v:.2f}"
            for col, v in row.items()
        ]
        rows.append(f"{idx} & " + " & ".join(vals) + r" \\")
    return rows

def convert_panels_to_latex(
    df_q: pd.DataFrame,
    df_a: pd.DataFrame,
    caption: str,
    label: str,
    note: str,
) -> str:
    """
    Build a *single* LaTeX table with two panels (Quarterly & Annual).
    Mean / StdDev already annualised upstream.
    """
    note = note.replace("%", "\\%")  # escape percent
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\begin{tiny}",
        rf"\caption{{\normalsize{{{caption}}}\\ \footnotesize{{{note}}}}}",
        rf"\label{{{label}}}",
    ]

    # ─── Panel A (Quarterly) ──────────────────────────────────────────────
    lines += [
        r"\textbf{Panel A. Quarterly Asset‑Class Returns (annualised)}\par\smallskip",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc}",
        r"\toprule",
        " & " + " & ".join(df_q.columns) + r" \\",
        r"\midrule",
        *df_to_latex_rows(df_q),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{1em}",
    ]

    # ─── Panel B (Annual) ────────────────────────────────────────────────
    lines += [
        r"\textbf{Panel B. Annual Endowment Returns}\par\smallskip",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc}",
        r"\toprule",
        " & " + " & ".join(df_a.columns) + r" \\",
        r"\midrule",
        *df_to_latex_rows(df_a),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{tiny}",
        r"\end{table}",
    ]
    return "\n".join(lines)


###############################################################################
#            LOAD  HEDGING STRATEGIES  (MONTHLY)  – unchanged
###############################################################################
def load_hedging_strategies_for_stats() -> pd.DataFrame:
    fp = "data/hedging_strategies.csv"
    df = pd.read_csv(fp, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df.set_index("Date", inplace=True)
    df.index = df.index + pd.offsets.MonthEnd(0)

    for col in df.columns:
        df[col] = (
            df[col]
            .str.rstrip("%")
            .replace("", np.nan)
            .astype(float)
            .div(100)
        )
    df.sort_index(inplace=True)
    df.rename(
        columns={
            "V Fast": "TS Momentum (Very Fast)",
            "Fast": "TS Momentum (Fast)",
            "Med": "TS Momentum (Medium)",
            "Slow": "TS Momentum (Slow)",
            "V Slow": "TS Momentum (Very Slow)",
        },
        inplace=True,
    )
    return df

def truncate_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc["1999-07-01":"2024-06-30"].copy()

###############################################################################
#                                 MAIN APP
###############################################################################
def main() -> None:
    st.title("Descriptive Statistics Dashboard")

    # 1  load data
    allocations, ret_q = load_and_preprocess_data()      # quarterly returns
    endow_ann = load_individual_endowments()             # annual

    # 2  strict‑overlap quarterly sample
    q_shared = get_longest_shared_period_strict(ret_q)
    if q_shared.empty:
        st.error("No overlapping quarterly range.")
        return
    desc_q = compute_descriptive_stats(q_shared, ann_factor=4)

    # 3  strict‑overlap annual endowment sample
    a_shared = get_longest_shared_period_strict(endow_ann)
    if a_shared.empty:
        st.error("No overlapping annual range.")
        return
    desc_a = compute_descriptive_stats(a_shared, ann_factor=1)

    # 4  display merged Table 1
    st.subheader("Table 1 – Descriptive Statistics")
    st.dataframe(
        pd.concat({"Quarterly (ann.)": desc_q, "Annual": desc_a},
                  axis=0,
                  names=["Panel", "Series"])
        .style.format("{:.2f}")
    )

    latex_tbl1 = convert_panels_to_latex(
        desc_q,
        desc_a,
        caption="Descriptive Statistics of Returns",
        label="tab:descriptive_stats",
        note="Mean and StdDev are annualised where applicable; Skewness and Excess Kurtosis are unit‑less.",
    )
    st.code(latex_tbl1, language="latex")

    # 5  hedging‑strategy table (unchanged from earlier version)
    st.subheader("Hedging Strategy Descriptive Stats (Table A‑1)")
    hedging_df = truncate_period(load_hedging_strategies_for_stats())
    hs_stats = compute_descriptive_stats(hedging_df, ann_factor=12)
    st.dataframe(hs_stats.style.format("{:.2f}"))

    st.code(
        convert_panels_to_latex(
            hs_stats, pd.DataFrame(),      # second panel empty → ignored
            caption="Hedging Strategies – Descriptive Stats",
            label="table:hedging_stats",
            note="Monthly data 1999‑07‑01 to 2024‑06‑30. Mean & StdDev annualised.",
        ),
        language="latex",
    )

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
