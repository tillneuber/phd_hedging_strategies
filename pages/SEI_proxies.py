import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

st.set_page_config(page_title="SEI vs. Listed-Proxy SEI", layout="wide")

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
    alloc_cols = [
        'Year', 'Public Equity', 'PE/VC', 'Hedge Funds',
        'Real Assets & ILBs', 'Fixed Income', 'Private Credit', 'Cash'
    ]
    allocations = pd.read_csv('data/hist_endowment_saa.csv', sep=';', names=alloc_cols, header=0)

    # wide→long
    allocations = allocations.melt(
        id_vars=['Year'], var_name='Asset Class', value_name='Allocation'
    )
    allocations['Allocation'] = pd.to_numeric(allocations['Allocation'], errors='coerce')
    allocations['Start Date'] = pd.to_datetime(allocations['Year'].astype(str)) + pd.DateOffset(months=6)
    allocations['End Date'] = allocations['Start Date'] + pd.DateOffset(years=1) - pd.DateOffset(days=1)

    # quarterly returns
    returns_quarterly = pd.read_csv('data/quarterly_returns.csv', sep=';', header=0)
    returns_quarterly['Date'] = pd.to_datetime(returns_quarterly['Date'], format='%d.%m.%Y', errors='coerce')
    returns_quarterly.set_index('Date', inplace=True)
    returns_quarterly.index = returns_quarterly.index + pd.offsets.MonthEnd(0)

    # Convert string percentages to decimal
    returns_quarterly = returns_quarterly.apply(
        lambda col: col.map(
            lambda x: float(str(x).replace('%', '')) / 100 if pd.notnull(x) and str(x) != '' else np.nan
        )
    )

    # Add a zero-return baseline row at 01‑Jul‑1999 if missing
    base_dt = pd.Timestamp("1999-07-01")
    if base_dt not in returns_quarterly.index:
        zero_vals = {col: 0.0 for col in returns_quarterly.columns}
        returns_quarterly.loc[base_dt] = zero_vals
    returns_quarterly.sort_index(inplace=True)

    return allocations, returns_quarterly

def estimate_lambda_ar1(rep_series: pd.Series) -> float:
    """
    Estimate lambda for the appraisal-smoothing model:
        r_rep_t = lambda * r_true_t + (1 - lambda) * r_rep_{t-1}.
    Under the reduced-form AR(1) on reported returns:
        r_rep_t = phi * r_rep_{t-1} + u_t, so lambda = 1 - phi.
    We estimate phi with OLS *without* intercept:
        phi_hat = sum(y_t * x_t) / sum(x_t^2), with x_t = r_{t-1}, y_t = r_t.

    Parameters
    ----------
    rep_series : pd.Series (quarterly returns in decimals)

    Returns
    -------
    float : lambda_hat = 1 - phi_hat
    """
    s = rep_series.astype(float).dropna()
    if len(s) < 3:
        return np.nan  # not enough data points for AR(1)

    y = s.iloc[1:].values
    x = s.shift(1).iloc[1:].values

    denom = np.dot(x, x)
    if denom == 0:
        return np.nan

    phi_hat = float(np.dot(x, y) / denom)
    lambda_hat = 1.0 - phi_hat
    return lambda_hat

def approximate_quarters_diff(ts_start, ts_end):
    """Rounded # of quarters between two timestamps."""
    days_diff = (ts_end - ts_start).days
    months = days_diff / 30.4375
    q_float = months / 3.0
    q_rounded = int(round(q_float))
    return max(1, q_rounded)

def find_time_to_recovery_q(cum_series, start_d, trough_d):
    """
    Integer # of quarters from the trough until the series returns to or exceeds
    its value at 'start_d'. Returns None if not recovered in-sample.
    """
    if start_d not in cum_series.index or trough_d not in cum_series.index:
        return None
    peak_val = cum_series.loc[start_d]
    subset = cum_series.loc[trough_d:]
    rec_idx = subset[subset >= peak_val].index
    if len(rec_idx) == 0:
        return None
    # round to nearest quarter
    return approximate_quarters_diff(trough_d, rec_idx[0])

def pivot_crises_into_columns_generic(crises, cum_dict, primary_key):
    """
    Build a crisis table (columns = Crisis1..N) for the given 'primary_key'
    series using the same structure/calcs as your main page:
      • Beginning, Trough, Recovery dates
      • Max Drawdown
      • Drawdown Length (# quarters)
      • Time to Recovery (# quarters)
      • Public Equity Drawdown & Time to Recovery (for the same windows)

    Parameters
    ----------
    crises : list of dicts from find_crisis_periods_for_rep_endowment(...)
    cum_dict : dict[str, pd.Series] with cumulative series. Must include:
               primary_key, and optionally "Public Equity".
    primary_key : str — which series in cum_dict defines the crisis windows.

    Returns
    -------
    pd.DataFrame
    """
    if not crises:
        return pd.DataFrame()

    col_names = [f"Crisis{i+1}" for i in range(len(crises))]
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

    prim_cum = cum_dict.get(primary_key, pd.Series(dtype=float))
    pe_cum   = cum_dict.get("Public Equity", pd.Series(dtype=float))

    for i, crisis in enumerate(crises):
        cname = col_names[i]
        start_dt = crisis['Start']
        trough_dt = crisis['Trough']
        end_dt = crisis['End']
        max_dd = crisis['Max Drawdown']

        data[cname][0] = str(start_dt.date())             # Beginning
        data[cname][1] = str(trough_dt.date())            # Trough
        data[cname][2] = str(end_dt.date())               # Recovery
        data[cname][3] = f"{max_dd:.1%}"                  # Max DD
        data[cname][4] = f"{approximate_quarters_diff(start_dt, trough_dt)}"
        ttr = find_time_to_recovery_q(prim_cum, start_dt, trough_dt)
        data[cname][5] = f"{ttr}" if ttr is not None else "n/a"
        data[cname][6] = "---"

        # Public Equity comparison for the same start→trough window
        if not pe_cum.empty and (start_dt in pe_cum.index) and (trough_dt in pe_cum.index):
            pk_val = pe_cum.loc[start_dt]
            th_val = pe_cum.loc[trough_dt]
            decline_e = (th_val - pk_val) / pk_val
            data[cname][7] = "n/a" if decline_e > -1e-9 else f"{decline_e:.1%}"
            pe_ttr = find_time_to_recovery_q(pe_cum, start_dt, trough_dt)
            data[cname][8] = f"{pe_ttr}" if pe_ttr is not None else "n/a"
        else:
            data[cname][7] = "n/a"
            data[cname][8] = "n/a"

    return pd.DataFrame(data, index=row_labels)


def unsmooth_series(rep_series: pd.Series, lam: float) -> pd.Series:
    """
    Invert the appraisal smoothing:
        r_true_hat_t = (r_rep_t - (1 - lambda) * r_rep_{t-1}) / lambda

    Boundary fix: set the first unsmoothed observation equal to the reported
    return so the portfolio stays fully invested in the first quarter.
    """
    r = rep_series.astype(float)
    u = (r - (1.0 - lam) * r.shift(1)) / lam
    if len(u) and pd.isna(u.iloc[0]):
        u.iloc[0] = r.iloc[0]  # keep t0 invested
    return u


def unify_timeframe(allocations, returns_q):
    """
    Align timeframe for allocations & returns AND
    resample returns to quarter-end (QE-DEC) using compounded returns.

    - allocations: long-format SAA with Start/End Date (annual FY targets)
    - returns_q: monthly (end-of-month) or mixed-frequency returns, decimal

    Returns:
      allocations_q  -> filtered to [q_start, q_end]
      returns_q_qe   -> quarter-end (QE-DEC) compounded returns within each quarter
      q_start, q_end -> final overlapping window
    """

    # 1) Determine overlapping window on the raw inputs
    earliest_alloc_start = allocations['Start Date'].min()
    latest_alloc_end = allocations['End Date'].max()
    earliest_ret = returns_q.index.min()
    latest_ret = returns_q.index.max()

    raw_start = max(earliest_alloc_start, earliest_ret)
    raw_end   = min(latest_alloc_end, latest_ret)

    # Guard: trim both to this raw overlapping slice first
    mask_alloc = (allocations['End Date'] >= raw_start) & (allocations['Start Date'] <= raw_end)
    allocations = allocations[mask_alloc].copy()

    mask_rets = (returns_q.index >= raw_start) & (returns_q.index <= raw_end)
    returns_q = returns_q.loc[mask_rets].copy()

    # 2) Resample returns to quarter-end (QE-DEC) with compounding
    #    For each column, (1+r).prod()-1 within the quarter
    returns_q_qe = (
        (1.0 + returns_q)
        .resample('Q-DEC')
        .prod()
        .apply(lambda s: s - 1.0)
    )

    # 3) Final overlapping window after resampling
    q_start = max(allocations['Start Date'].min(), returns_q_qe.index.min())
    q_end   = min(allocations['End Date'].max(), returns_q_qe.index.max())

    # 4) Filter again to the final window
    mask_alloc_final = (allocations['End Date'] >= q_start) & (allocations['Start Date'] <= q_end)
    allocations_q = allocations[mask_alloc_final].copy()

    returns_q_qe = returns_q_qe.loc[(returns_q_qe.index >= q_start) & (returns_q_qe.index <= q_end)].copy()

    return allocations_q, returns_q_qe, q_start, q_end

def make_unsmoothed_returns(returns_q_qe: pd.DataFrame):
    """
    Create an unsmoothed (AR(1)-inverted) version of the quarterly SEI input returns
    for the three appraisal-based series:
        'PE/VC', 'Real Assets & ILBs', 'Private Credit'

    Parameters
    ----------
    returns_q_qe : pd.DataFrame
        Quarter-end (QE-DEC) compounded returns in decimals, columns include
        at least the three series named above.

    Returns
    -------
    returns_unsmoothed : pd.DataFrame
        Same shape as input; three target columns replaced by unsmoothed series.
    lambdas_dict       : dict
        {'PE/VC': lambda_hat, 'Real Assets & ILBs': lambda_hat, 'Private Credit': lambda_hat}
        (Can be displayed on-page.)
    """
    target_cols = ['PE/VC', 'Real Assets & ILBs', 'Private Credit']
    returns_unsmoothed = returns_q_qe.copy()
    lambdas = {}

    for col in target_cols:
        if col not in returns_q_qe.columns:
            continue
        lam = estimate_lambda_ar1(returns_q_qe[col])
        lambdas[col] = lam
        if pd.notnull(lam) and lam != 0:
            returns_unsmoothed[col] = unsmooth_series(returns_q_qe[col], lam)
        else:
            # if lambda cannot be estimated, leave reported series as-is
            returns_unsmoothed[col] = returns_q_qe[col]

    return returns_unsmoothed, lambdas

def build_lambda_table(lambdas_dict: dict) -> pd.DataFrame:
    """
    Turn the lambdas dict into a neat one-column DataFrame for display.
    """
    df = pd.DataFrame.from_dict(lambdas_dict, orient='index', columns=['Lambda (λ)'])
    # nice order
    desired = ['PE/VC', 'Private Credit', 'Real Assets & ILBs']
    ordered = [c for c in desired if c in df.index] + [i for i in df.index if i not in desired]
    return df.loc[ordered]

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
#  CRISIS DETECTION (≥ 10 %) QUARTERLY
########################################
def find_crisis_periods_for_rep_endowment(dd_series, threshold=0.10):
    """
    Identify crisis periods in the SEI quarterly drawdown if ≥ threshold (10 %).
    """
    ds = dd_series.copy()
    in_crisis = False
    crises = []
    start_date, trough_date, max_dd = None, None, 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
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


def create_performance_chart_jof_matplotlib(cum_dict, dd_dict, rep_crises=None):
    """
    Two-subplot figure: top=cumulative, bottom=drawdowns.
    Crisis intervals (from the baseline SEI) are shaded in light-grey.

    Expects possible keys in cum_dict/dd_dict:
      - "SEI (Reported NAVs)"
      - "SEI (Listed Proxies)"
      - "SEI (Unsmooth AR1)"
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9.5, 5.4))

    # Shade crisis intervals
    if rep_crises:
        for crisis in rep_crises:
            start, end = crisis["Start"], crisis["End"]
            ax1.axvspan(start, end, color="lightgray", alpha=0.3, zorder=0)
            ax2.axvspan(start, end, color="lightgray", alpha=0.3, zorder=0)

    # desired legend / plotting order
    desired_order = [
        "SEI (Reported NAVs)",
        "SEI (Listed Proxies)",
        "SEI (Unsmooth AR1)"
    ]

    # styling
    def style_for(label):
        if label == "SEI (Reported NAVs)":
            return dict(color="black", linestyle="solid", linewidth=1.4, zorder=3)
        if label == "SEI (Listed Proxies)":
            return dict(color="tab:blue", linestyle="dashed", linewidth=1.4, zorder=2)
        if label == "SEI (Unsmooth AR1)":
            return dict(color="tab:red", linestyle="dashdot", linewidth=1.4, zorder=2)
        return dict(color="gray", linestyle="solid", linewidth=1.2, zorder=1)

    # TOP: cumulative
    for label in desired_order:
        s = cum_dict.get(label, pd.Series(dtype=float))
        if not s.empty:
            ax1.plot(s.index, s.values, label=label, **style_for(label))
    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel("Cumulative Returns", fontsize=9)
    ax1.tick_params(labelsize=8)

    # BOTTOM: drawdowns
    for label in desired_order:
        d = dd_dict.get(label, pd.Series(dtype=float))
        if not d.empty:
            ax2.plot(d.index, d.values, label=label, **style_for(label))
    ax2.set_ylabel("Drawdowns", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    # x-axis
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(pad=1.2)
    return fig

def main():
    st.title("SEI vs. Listed-Proxy SEI (Quarterly)")

    st.write(
        "This page compares the **baseline Synthetic Endowment Index (SEI)**—which uses "
        "**PE/VC**, **Real Assets & ILBs**, and **Private Credit** appraisal-based series—"
        "to a **listed version** (Listed Private Equity, REITs, BDCs) and an "
        "**unsmoothed AR(1)** version of the private sleeves. "
        "Crises and tables use the same methodology as on the main page."
    )

    # 1) Load & align data (returns resampled to QE-DEC inside unify_timeframe)
    allocations, ret_q_full = load_and_preprocess_data()
    allocations, ret_q, q_start, q_end = unify_timeframe(allocations, ret_q_full.copy())

    # Quarterly index for mapping SAA to quarters
    idx_q = pd.date_range(q_start, q_end, freq='QE-DEC')

    # 2) Map allocations to quarterly dates
    hist_alloc_q = map_allocations_to_periods(allocations, idx_q)

    # 3) Build returns: baseline, listed proxies, unsmoothed AR(1)
    baseline_cols = [
        'Public Equity', 'PE/VC', 'Hedge Funds',
        'Real Assets & ILBs', 'Fixed Income', 'Private Credit', 'Cash'
    ]
    proxy_map = {
        'PE/VC': 'Listed Private Equity',
        'Real Assets & ILBs': 'REITs',
        'Private Credit': 'BDCs'
    }

    # Align allocations to available return columns
    common_baseline = [c for c in baseline_cols if c in ret_q.columns]
    alloc_base = hist_alloc_q.reindex(columns=common_baseline).fillna(0)
    rets_base = ret_q.reindex(columns=common_baseline)

    # Listed-proxy returns
    rets_listed = rets_base.copy()
    for base_col, proxy_col in proxy_map.items():
        if base_col in rets_listed.columns and proxy_col in ret_q.columns:
            rets_listed[base_col] = ret_q[proxy_col]

    # Unsmoothed returns for the 3 appraisal-based sleeves + λ table
    rets_unsmoothed, lambdas = make_unsmoothed_returns(ret_q)

    # Weighted sum helper (quarterly)
    def weighted_sum_returns(weights_df, returns_df):
        aligned_w = weights_df.reindex(returns_df.index, method='ffill')
        common_cols = returns_df.columns.intersection(aligned_w.columns)
        aligned_r = returns_df[common_cols].copy()
        aligned_w = aligned_w[common_cols].copy()
        aligned_w = aligned_w.where(aligned_r.notna(), 0.0)
        aligned_r = aligned_r.fillna(0.0)
        return (aligned_r * aligned_w).sum(axis=1)

    # 4) Portfolio returns
    sei_ret            = weighted_sum_returns(alloc_base, rets_base)
    sei_listed_ret     = weighted_sum_returns(alloc_base, rets_listed)
    sei_unsmoothed_ret = weighted_sum_returns(alloc_base, rets_unsmoothed)

    # 5) Cumulative & drawdowns
    sei_cum, sei_dd                         = calculate_cumulative_and_dd(sei_ret)
    sei_listed_cum, sei_listed_dd           = calculate_cumulative_and_dd(sei_listed_ret)
    sei_unsmoothed_cum, sei_unsmoothed_dd   = calculate_cumulative_and_dd(sei_unsmoothed_ret)

    # Public Equity cumulative for comparison rows in the crisis tables
    pe_cum = pd.Series(dtype=float)
    if 'Public Equity' in ret_q.columns:
        pe_q = ret_q['Public Equity'].dropna()
        pe_cum, _ = calculate_cumulative_and_dd(pe_q)

    # 6) λ display
    st.subheader("AR(1) λ Estimates (Quarterly)")
    lambda_table = build_lambda_table(lambdas)
    st.dataframe(lambda_table.style.format({"Lambda (λ)": "{:.3f}"}))

    # 7) Chart
    st.subheader("Matplotlib Journal-Style Performance Chart")
    st.caption("Shaded bands mark ≥10% drawdown crises detected from the baseline SEI.")
    rep_crises_q = find_crisis_periods_for_rep_endowment(sei_dd, threshold=0.10)
    cum_dict = {
        "SEI (Reported NAVs)": sei_cum,
        "SEI (Listed Proxies)": sei_listed_cum,
        "SEI (Unsmooth AR1)":  sei_unsmoothed_cum,
    }
    dd_dict = {
        "SEI (Reported NAVs)": sei_dd,
        "SEI (Listed Proxies)": sei_listed_dd,
        "SEI (Unsmooth AR1)":  sei_unsmoothed_dd,
    }
    fig = create_performance_chart_jof_matplotlib(cum_dict, dd_dict, rep_crises=rep_crises_q)
    st.pyplot(fig, use_container_width=True)

    # 8) Summary stats
    st.subheader("Summary Statistics (Quarterly)")
    def stats_from_returns(r):
        ann_mean = r.mean() * 4
        ann_vol  = r.std(ddof=0) * np.sqrt(4)
        sharpe   = ann_mean / ann_vol if ann_vol and ann_vol > 0 else np.nan
        total_cum = (1 + r).prod() - 1
        return pd.Series({
            "Ann. Mean": ann_mean,
            "Ann. Vol": ann_vol,
            "Sharpe (rf≈0)": sharpe,
            "Total Cum.": total_cum
        })
    summary_df = pd.DataFrame({
        "SEI (Reported NAVs)": stats_from_returns(sei_ret),
        "SEI (Listed Proxies)": stats_from_returns(sei_listed_ret),
        "SEI (Unsmooth AR1)":  stats_from_returns(sei_unsmoothed_ret),
    }).T
    st.dataframe(summary_df.style.format({
        "Ann. Mean": "{:.2%}",
        "Ann. Vol": "{:.2%}",
        "Sharpe (rf≈0)": "{:.2f}",
        "Total Cum.": "{:.2%}"
    }))

    # 9) Crisis tables for each series (same methodology as original)
    st.subheader("Table 1: Quarterly Crisis Periods (≥ 10 % Drawdown) — SEI (Reported NAVs)")
    crises_rep = find_crisis_periods_for_rep_endowment(sei_dd, threshold=0.10)
    table_rep = pivot_crises_into_columns_generic(
        crises_rep,
        {"SEI (Reported NAVs)": sei_cum, "Public Equity": pe_cum},
        primary_key="SEI (Reported NAVs)"
    )
    st.dataframe(table_rep)

    st.subheader("Table 2: Quarterly Crisis Periods (≥ 10 % Drawdown) — SEI (Unsmooth AR1)")
    crises_uns = find_crisis_periods_for_rep_endowment(sei_unsmoothed_dd, threshold=0.10)
    table_uns = pivot_crises_into_columns_generic(
        crises_uns,
        {"SEI (Unsmooth AR1)": sei_unsmoothed_cum, "Public Equity": pe_cum},
        primary_key="SEI (Unsmooth AR1)"
    )
    st.dataframe(table_uns)

    st.subheader("Table 3: Quarterly Crisis Periods (≥ 10 % Drawdown) — SEI (Listed Proxies)")
    crises_list = find_crisis_periods_for_rep_endowment(sei_listed_dd, threshold=0.10)
    table_listed = pivot_crises_into_columns_generic(
        crises_list,
        {"SEI (Listed Proxies)": sei_listed_cum, "Public Equity": pe_cum},
        primary_key="SEI (Listed Proxies)"
    )
    st.dataframe(table_listed)



if __name__ == '__main__':
    main()
