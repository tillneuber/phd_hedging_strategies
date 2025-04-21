import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# ──────────────────────────────────────────────────────────────────────
# I.  DATA HELPERS
# ──────────────────────────────────────────────────────────────────────
def load_endowment_allocations_and_returns():
    alloc_cols = ["Year","Public Equity","PE/VC","Hedge Funds",
                  "Real Assets & ILBs","Fixed Income","Private Credit","Cash"]
    alloc = pd.read_csv("data/hist_endowment_saa.csv",sep=";",header=0,names=alloc_cols)
    alloc_long = alloc.melt(id_vars=["Year"],var_name="Asset Class",
                            value_name="Allocation").astype({"Allocation":float})
    alloc_long["Start"] = pd.to_datetime(alloc_long["Year"].astype(str))+pd.DateOffset(months=6)
    alloc_long["End"]   = alloc_long["Start"]+pd.DateOffset(years=1)-pd.DateOffset(days=1)

    ret_q = pd.read_csv("data/quarterly_returns.csv",sep=";",header=0)
    ret_q["Date"] = pd.to_datetime(ret_q["Date"],format="%d.%m.%Y",errors="coerce")
    ret_q.set_index("Date",inplace=True)
    ret_q.index = ret_q.index + pd.offsets.MonthEnd(0)
    for c in ret_q.columns:
        if ret_q[c].dtype=="O":
            ret_q[c]=ret_q[c].str.replace("%","",regex=False).replace("",np.nan).astype(float)/100
    anchor=pd.Timestamp("1999-07-31")
    if anchor not in ret_q.index:
        ret_q.loc[anchor]=0.0
    ret_q.sort_index(inplace=True)
    return alloc_long, ret_q


def unify_timeframe(alloc, ret):
    start=max(alloc["Start"].min(),ret.index.min())
    end  =min(alloc["End"].max(), ret.index.max())
    return alloc[(alloc["End"]>=start)&(alloc["Start"]<=end)], ret.loc[start:end], start, end


def map_alloc(alloc_long, idx_q):
    out=pd.DataFrame(index=idx_q)
    for asset in alloc_long["Asset Class"].unique():
        sub=alloc_long[alloc_long["Asset Class"]==asset]
        for _,r in sub.iterrows():
            rng=pd.date_range(r["Start"],r["End"],freq=idx_q.freq)
            out.loc[out.index.intersection(rng),asset]=r["Allocation"]
    return out.fillna(0.0)


def calc_cum_dd(r):
    cum=(1+r).cumprod()
    dd=cum.div(cum.cummax())-1
    return cum, dd


def load_hedges():
    df = pd.read_csv("data/hedging_strategies.csv", sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df.set_index("Date", inplace=True)
    df.index += pd.offsets.MonthEnd(0)

    for col in df.columns:
        # Ensure that missing data points are treated as NaN and not 0
        df[col] = df[col].apply(lambda x: pd.to_numeric(x.replace("%", ""), errors='coerce') / 100 if pd.notnull(x) else np.nan)
    
    df.sort_index(inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# II.  CRISIS LOGIC
# ──────────────────────────────────────────────────────────────────────
def detect_crises(dd, thresh):
    dd=dd.dropna()
    crises,in_c=False,False
    start=trough=maxdd=None
    out=[]
    for d,v in dd.items():
        if not in_c and v<=-thresh:
            in_c=True
            s_prev=dd.loc[:d][dd.loc[:d]==0].last_valid_index()
            start=s_prev if s_prev is not None else d
            trough=d; maxdd=v
        if in_c:
            if v<maxdd: maxdd=v; trough=d
            if v==0:
                out.append({"Start":start,"Trough":trough,"End":d,"Max":maxdd})
                in_c=False
    if in_c:
        out.append({"Start":start,"Trough":trough,"End":dd.index[-1],"Max":maxdd})
    return out


def peak_to_trough(cum, s, t):
    """
    Calculate the peak-to-trough return for the hedge strategy during the crisis.
    If the crisis time period is not fully covered by the hedge data, return NaN ("NA").
    """
    # Ensure that both the start (s) and trough (t) are present in the cumulative return series
    if (s not in cum.index) or (t not in cum.index):
        return np.nan  # Return NaN if the crisis period isn't fully covered by the hedge data
    
    return cum.loc[t] / cum.loc[s] - 1


def quarters(a,b):
    return max(int(round((b-a).days/91.3125)),1)


# ──────────────────────────────────────────────────────────────────────
# III.  BUILD TABLE
# ──────────────────────────────────────────────────────────────────────
def build_crisis_table(crises, cum_hedge_dict):
    """
    Builds a crisis table with the hedge performance (peak-to-trough) for each crisis.
    If hedge strategy data is not available for the full crisis period, it will show "NA".
    """
    if not crises:
        return pd.DataFrame()

    hedge_cols = list(cum_hedge_dict.keys())
    rows = []

    for i, c in enumerate(crises, start=1):
        s_dt, t_dt, e_dt, max_dd = c["Start"], c["Trough"], c["End"], c["Max"]

        row = {
            "#": i,
            "Start": s_dt.date(),
            "Trough": t_dt.date(),
            "End": e_dt.date(),
            "Max DD": f"{max_dd:.1%}",
            "Depth (Q)": quarters(s_dt, t_dt),
            "To Rec (Q)": quarters(t_dt, e_dt),
        }

        # ▸ hedge peak‑to‑trough returns (with coverage check)
        for hcol in hedge_cols:
            series = cum_hedge_dict[hcol]
            # Ensure the hedge data covers the entire crisis period
            if s_dt >= series.index.min() and t_dt <= series.index.max():  
                pt_val = peak_to_trough(series, s_dt, t_dt)
            else:
                pt_val = np.nan  # Set to NaN if the hedge data does not cover the crisis period
            row[hcol] = "NA" if pd.isna(pt_val) else f"{pt_val:.1%}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # ▸ summary rows (average and median)
    def make_summary_row(func, label):
        r = {k: "" for k in df.columns}
        r["#"], r["Max DD"] = label, f"--{label}--"
        for h in hedge_cols:
            pct = (
                df[h].replace("NA", np.nan)
                     .str.rstrip("%").astype(float) / 100
            )
            val = getattr(pct.dropna(), func)(skipna=True)
            r[h] = "NA" if pd.isna(val) else f"{val:.1%}"
        return r

    df = pd.concat(
        [df,
         pd.DataFrame([make_summary_row("mean", "ALL AVG"),
                       make_summary_row("median", "ALL MED")])],
        ignore_index=True,
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# IV. Convert a crisis table → LaTeX  (UPDATED only in one line)
# ──────────────────────────────────────────────────────────────────────
def to_latex(df, title, threshold):
    """
    Minimal LaTeX table.  Ensure missing values are shown as "NA" for better clarity.
    """
    align = "l" + "c" * (df.shape[1] - 1)
    caption = (
        rf"\caption{{\normalsize{{{title}}}\\" +
        rf"\footnotesize{{Drawdown threshold: {threshold:.1f}\%}}}}"
    )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\begin{tiny}",
        caption,
        rf"\label{{table:{title.lower().replace(' ','_')}}}",
        rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{align}}}",
        r"\toprule",
    ]
    # (rest of the function is unchanged)
    col_list = df.columns.tolist()
    lines.append(" & ".join([""] + col_list) + r" \\")
    lines.append(r"\midrule")
    
    # Replace NaN with 'NA' for missing data
    for _, row in df.iterrows():
        lines.append(" & ".join([str(row[c]).replace('nan', 'NA') for c in col_list]) + r" \\")
    
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{tiny}", r"\end{table}"]
    return "\n".join(lines)



# ──────────────────────────────────────────────────────────────────────
# V.  TSM‑MEDIAN CHART  (10 % threshold)
# ──────────────────────────────────────────────────────────────────────
TSM_MAP={4:"V Fast",7:"Fast",12:"Med",20:"Slow",24:"V Slow"}

def median_tsm_by_speed(dd_series, crises, cum_hedge):
    """
    returns a dict {speed: median_peak‑to‑trough_return}
    for the five TSM columns, given a list of crises for a portfolio.
    """
    out={}
    for w,col in TSM_MAP.items():
        vals=[]
        if col not in cum_hedge:        # column missing → skip
            out[w]=np.nan; continue
        for c in crises:
            vals.append(peak_to_trough(cum_hedge[col],c["Start"],c["Trough"]))
        out[w]=np.nan if not vals else float(np.median(vals))
    return out


# ──────────────────────────────────────────────────────────────────────
# Extra helper for the TSM chart   (NEW / REPLACEMENT)
# ──────────────────────────────────────────────────────────────────────
# helper → fixed mapping from speed (weeks) → column name in hedges CSV
# ---------------------------------------------------------------------
def trend_sensitivity():
    """
    Return the list of trend speeds (in weeks) that correspond to the
    five TSM columns inside hedging_strategies.csv.
          Very Fast = 4 w,  Fast = 7 w,  Medium = 12 w,
          Slow = 20 w,      Very Slow = 24 w
    """
    return [4, 7, 12, 20, 24]

def compute_tsm_medians(cum_hedge_dict, crises_dict, trend_sens):
    """
    Parameters
    ----------
    cum_hedge_dict : dict
        {strategy_name : cumulative_index_series}
        (already built in main from hedge_q)
        must contain the five TSM columns:
        'V Fast','Fast','Med','Slow','V Slow'
    crises_dict : dict
        {"PuE": [list_of_crisis_dicts],  "SEI": [...] }
        crisis_dict items are exactly those you pass to build_crisis_table
    trend_sens : list[int]
        [ 4, 7, 12, 20, 24 ]
    Returns
    -------
    pe_median_list , sei_median_list, pe_mean_list, sei_mean_list
        four lists of the same length as trend_sens
        NaN where a given TSM column is unavailable
    """
    speed_to_col = {4: "V Fast",
                    7: "Fast",
                    12: "Med",
                    20: "Slow",
                    24: "V Slow"}

    def series_stats(crises, col_name):
        if col_name not in cum_hedge_dict:
            return np.nan, np.nan
        cum = cum_hedge_dict[col_name]
        vals = [peak_to_trough(cum, c["Start"], c["Trough"]) for c in crises]
        # filter NaN & None
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            return np.nan, np.nan
        return float(np.nanmedian(vals)), float(np.nanmean(vals))

    pe_med, sei_med, pe_mean, sei_mean = [], [], [], []
    for spd in trend_sens:
        col = speed_to_col[spd]
        pe_m, pe_mu = series_stats(crises_dict["PuE"],  col)
        sei_m, sei_mu = series_stats(crises_dict["SEI"], col)
        pe_med.append(pe_m)
        sei_med.append(sei_m)
        pe_mean.append(pe_mu)
        sei_mean.append(sei_mu)

    return pe_med, sei_med, pe_mean, sei_mean


# ──────────────────────────────────────────────────────────────────────
# V.  MAIN PAGE
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# V.  Main Streamlit page  (FULLY REWRITTEN)
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
def main():
    st.title("Robustness Checks — thresholds, largest drawdowns & TSM chart")

    # 1. Define the trend sensitivity
    trend_sens = trend_sensitivity()  # [4, 7, 12, 20, 24]

    # 2. Load data
    alloc_long, ret_q = load_endowment_allocations_and_returns()
    alloc_long, ret_q, start_dt, end_dt = unify_timeframe(alloc_long, ret_q)

    idx_q = pd.date_range(start_dt, end_dt, freq="Q")

    # Synthetic Endowment & Public-Equity returns
    weights_df = map_alloc(alloc_long, idx_q)
    sei_r = (weights_df[weights_df.columns.intersection(ret_q.columns)] *
             ret_q).sum(axis=1)
    pe_r  = ret_q["Public Equity"].reindex(idx_q).fillna(0.0)

    cum_sei, dd_sei = calc_cum_dd(sei_r)
    cum_pe,  dd_pe  = calc_cum_dd(pe_r)

    # 3. Load hedges
    hedge_q = load_hedges().resample("Q").last().reindex(idx_q).fillna(0.0)
    cum_hedge = {c: calc_cum_dd(ser.dropna())[0]
                 for c, ser in hedge_q.items() if ser.dropna().size > 1}

    # ----- 10 % crisis tables -----
    thr = 0.10
    crises_pe  = detect_crises(dd_pe,  thr)
    crises_sei = detect_crises(dd_sei, thr)
    for name, crises in [("Public Equity", crises_pe),
                         ("Synthetic Endowment", crises_sei)]:
        st.header(f"{name} — 10 % threshold")
        df = build_crisis_table(crises, cum_hedge)
        st.dataframe(df, height=350)
        st.code(to_latex(df, f"{name} Crises 10%", 10.0), language="latex")

    # ----- 10 % Threshold plot -----
    st.header("10% Threshold Crisis Response")
    pe_med_10, sei_med_10, pe_mean_10, sei_mean_10 = compute_tsm_medians(
        cum_hedge,
        {"PuE": crises_pe, "SEI": crises_sei},
        trend_sens
    )
    
    fig_10, ax_10 = plt.subplots(figsize=(7.5, 5))
    ax_10.plot(trend_sens, pe_med_10,  marker="o", label="Public Equity (Median)", color="tab:blue")
    ax_10.plot(trend_sens, sei_med_10, marker="o", label="Synthetic Endowment (Median)", color="tab:orange")
    ax_10.plot(trend_sens, pe_mean_10, marker="x", linestyle="--", label="Public Equity (Mean)", color="tab:blue")
    ax_10.plot(trend_sens, sei_mean_10, marker="x", linestyle="--", label="Synthetic Endowment (Mean)", color="tab:orange")
    ax_10.set_xlabel("Trend Sensitivity (weeks)")
    ax_10.set_ylabel("Crisis Response (Peak–Trough Return)")
    ax_10.set_title("TSM Hedge Efficacy for 10% Threshold")
    ax_10.set_xticks(trend_sens)
    ax_10.grid(True, ls="--", lw=0.5)
    ax_10.legend()
    st.pyplot(fig_10)

    # ----- 7.5% threshold tables -----
    st.header("Public Equity and Synthetic Endowment — 7.5% Threshold")
    thr = 0.075
    crises_pe_75  = detect_crises(dd_pe,  thr)
    crises_sei_75 = detect_crises(dd_sei, thr)
    for name, crises in [("Public Equity", crises_pe_75),
                         ("Synthetic Endowment", crises_sei_75)]:
        df_75 = build_crisis_table(crises, cum_hedge)
        st.dataframe(df_75, height=350)
        st.code(to_latex(df_75, f"{name} Crises 7.5%", 7.5), language="latex")

    # ----- 7.5% Threshold plot -----
    st.header("7.5% Threshold Crisis Response")
    pe_med_75, sei_med_75, pe_mean_75, sei_mean_75 = compute_tsm_medians(
        cum_hedge,
        {"PuE": crises_pe_75, "SEI": crises_sei_75},
        trend_sens
    )
    
    fig_75, ax_75 = plt.subplots(figsize=(7.5, 5))
    ax_75.plot(trend_sens, pe_med_75,  marker="o", label="Public Equity (Median)", color="tab:blue")
    ax_75.plot(trend_sens, sei_med_75, marker="o", label="Synthetic Endowment (Median)", color="tab:orange")
    ax_75.plot(trend_sens, pe_mean_75, marker="x", linestyle="--", label="Public Equity (Mean)", color="tab:blue")
    ax_75.plot(trend_sens, sei_mean_75, marker="x", linestyle="--", label="Synthetic Endowment (Mean)", color="tab:orange")
    ax_75.set_xlabel("Trend Sensitivity (weeks)")
    ax_75.set_ylabel("Crisis Response (Peak–Trough Return)")
    ax_75.set_title("TSM Hedge Efficacy for 7.5% Threshold")
    ax_75.set_xticks(trend_sens)
    ax_75.grid(True, ls="--", lw=0.5)
    ax_75.legend()
    st.pyplot(fig_75)

    # ----- 12.5% threshold tables -----
    st.header("Public Equity and Synthetic Endowment — 12.5% Threshold")
    thr = 0.125
    crises_pe_125  = detect_crises(dd_pe,  thr)
    crises_sei_125 = detect_crises(dd_sei, thr)
    for name, crises in [("Public Equity", crises_pe_125),
                         ("Synthetic Endowment", crises_sei_125)]:
        df_125 = build_crisis_table(crises, cum_hedge)
        st.dataframe(df_125, height=350)
        st.code(to_latex(df_125, f"{name} Crises 12.5%", 12.5), language="latex")

    # ----- 12.5% Threshold plot -----
    st.header("12.5% Threshold Crisis Response")
    pe_med_125, sei_med_125, pe_mean_125, sei_mean_125 = compute_tsm_medians(
        cum_hedge,
        {"PuE": crises_pe_125, "SEI": crises_sei_125},
        trend_sens
    )
    
    fig_125, ax_125 = plt.subplots(figsize=(7.5, 5))
    ax_125.plot(trend_sens, pe_med_125,  marker="o", label="Public Equity (Median)", color="tab:blue")
    ax_125.plot(trend_sens, sei_med_125, marker="o", label="Synthetic Endowment (Median)", color="tab:orange")
    ax_125.plot(trend_sens, pe_mean_125, marker="x", linestyle="--", label="Public Equity (Mean)", color="tab:blue")
    ax_125.plot(trend_sens, sei_mean_125, marker="x", linestyle="--", label="Synthetic Endowment (Mean)", color="tab:orange")
    ax_125.set_xlabel("Trend Sensitivity (weeks)")
    ax_125.set_ylabel("Crisis Response (Peak–Trough Return)")
    ax_125.set_title("TSM Hedge Efficacy for 12.5% Threshold")
    ax_125.set_xticks(trend_sens)
    ax_125.grid(True, ls="--", lw=0.5)
    ax_125.legend()
    st.pyplot(fig_125)

    # ----- Top 4 Largest Crises tables -----
    st.header("Public Equity and Synthetic Endowment — Top 4 Largest Crises")
    top_4_pe  = sorted(crises_pe, key=lambda x: abs(x['Max']))[-4:]
    top_4_sei = sorted(crises_sei, key=lambda x: abs(x['Max']))[-4:]
    for name, crises in [("Public Equity", top_4_pe),
                         ("Synthetic Endowment", top_4_sei)]:
        df_top4 = build_crisis_table(crises, cum_hedge)
        st.dataframe(df_top4, height=350)
        st.code(to_latex(df_top4, f"{name} Top 4 Largest Crises", 0.0), language="latex")

    # ----- Top 4 Largest Crises plot -----
    st.header("Top 4 Largest Crises Crisis Response")
    pe_med_top4, sei_med_top4, pe_mean_top4, sei_mean_top4 = compute_tsm_medians(
        cum_hedge,
        {"PuE": top_4_pe, "SEI": top_4_sei},
        trend_sens
    )
    
    fig_top4, ax_top4 = plt.subplots(figsize=(7.5, 5))
    ax_top4.plot(trend_sens, pe_med_top4,  marker="o", label="Public Equity (Median)", color="tab:blue")
    ax_top4.plot(trend_sens, sei_med_top4, marker="o", label="Synthetic Endowment (Median)", color="tab:orange")
    ax_top4.plot(trend_sens, pe_mean_top4, marker="x", linestyle="--", label="Public Equity (Mean)", color="tab:blue")
    ax_top4.plot(trend_sens, sei_mean_top4, marker="x", linestyle="--", label="Synthetic Endowment (Mean)", color="tab:orange")
    ax_top4.set_xlabel("Trend Sensitivity (weeks)")
    ax_top4.set_ylabel("Crisis Response (Peak–Trough Return)")
    ax_top4.set_title("TSM Hedge Efficacy for Top 4 Largest Crises")
    ax_top4.set_xticks(trend_sens)
    ax_top4.grid(True, ls="--", lw=0.5)
    ax_top4.legend()
    st.pyplot(fig_top4)

if __name__ == "__main__":
    main()
