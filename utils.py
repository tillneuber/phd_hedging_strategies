# utils.py

import pandas as pd
import numpy as np

def load_asset_allocation_and_returns_data(
    alloc_file: str = "data/hist_endowment_saa.csv",
    returns_file: str = "data/quarterly_returns.csv"
) -> (pd.DataFrame, pd.DataFrame):
    """
    Load and preprocess endowment asset-allocation and quarterly-return data.
    Returns:
      - allocations: long-form DataFrame with 'Asset Class', 'Allocation', 'Start Date', 'End Date'
      - returns_q: quarterly returns (indexed by month-end) in decimal form
    """
    # --- allocations ---
    alloc_cols = [
        "Year", "Public Equity", "PE/VC", "Hedge Funds",
        "Real Assets & ILBs", "Fixed Income", "Private Credit", "Cash",
    ]
    alloc = pd.read_csv(alloc_file, sep=";", names=alloc_cols, header=0)
    alloc_long = (
        alloc
        .melt(id_vars=["Year"], var_name="Asset Class", value_name="Allocation")
        .assign(Allocation=lambda df: pd.to_numeric(df["Allocation"], errors="coerce"))
    )
    alloc_long["Start Date"] = pd.to_datetime(alloc_long["Year"].astype(str)) + pd.DateOffset(months=6)
    alloc_long["End Date"]   = alloc_long["Start Date"] + pd.DateOffset(years=1) - pd.DateOffset(days=1)

    # --- returns ---
    returns_q = pd.read_csv(returns_file, sep=";", header=0)
    returns_q["Date"] = pd.to_datetime(returns_q["Date"], format="%d.%m.%Y", errors="coerce")
    returns_q.set_index("Date", inplace=True)
    returns_q.index = returns_q.index + pd.offsets.MonthEnd(0)
    returns_q = returns_q.apply(
        lambda col: col.map(
            lambda x: float(str(x).replace("%",""))/100 if pd.notnull(x) else np.nan
        )
    )

    # add 0% baseline at 1999-07-01 if missing
    base = pd.Timestamp("1999-07-01")
    if base not in returns_q.index:
        returns_q.loc[base] = {c: 0.0 for c in returns_q.columns}
    returns_q.sort_index(inplace=True)

    return alloc_long, returns_q


def load_endowment_returns_data(
    file_path: str = "data/individual_endowments.csv"
) -> pd.DataFrame:
    """
    Load annual returns for Yale, Stanford, Harvard, NACUBO.
    Returns decimal returns, indexed by report date.
    """
    df = pd.read_csv(file_path, sep=";", header=0)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df.set_index("Date", inplace=True)
    df = df.applymap(lambda x: float(str(x).replace("%",""))/100 if pd.notnull(x) else np.nan)

    base = pd.Timestamp("1999-07-01")
    if base not in df.index:
        df.loc[base] = {c: 0.0 for c in df.columns}
    df.sort_index(inplace=True)
    return df


def load_hedge_strategies() -> pd.DataFrame:
    """
    Load the hedging strategy CSV, convert percentages to decimals,
    rename the five time-series-momentum sleeves, and create an equal-weight
    TSM Basket column.
    """
    file_path = "data/hedging_strategies.csv"
    df = pd.read_csv(file_path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df.set_index("Date", inplace=True)
    df.index = df.index + pd.offsets.MonthEnd(0)

    # convert % strings → decimal floats
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.rstrip("%")
            .replace("", np.nan)
            .astype(float)
            .div(100)
        )

    df.sort_index(inplace=True)

    # friendly column names
    rename_map = {
        "V Fast":  "Time Series Momentum (Very Fast)",
        "Fast":    "Time Series Momentum (Fast)",
        "Med":     "Time Series Momentum (Med)",
        "Slow":    "Time Series Momentum (Slow)",
        "V Slow":  "Time Series Momentum (Very Slow)",
    }
    df.rename(columns=rename_map, inplace=True)

    # add the equal-weight basket across the five sleeves
    tsm_cols = list(rename_map.values())
    df["Time Series Momentum (Basket)"] = df[tsm_cols].mean(axis=1)

    return df


def calculate_cumulative_returns_and_drawdowns(
    return_series: pd.Series
) -> (pd.Series, pd.Series):
    """
    From a returns series r_t, compute:
      cum_t = ∏_{s=1}^t (1 + r_s)
      dd_t  = (cum_t / max_{u ≤ t} cum_u) - 1
    """
    cum = (1 + return_series).cumprod()
    run_max = cum.cummax()
    dd = (cum - run_max) / run_max
    return cum, dd


def detect_crisis_periods(
    drawdown_series: pd.Series,
    threshold: float = 0.10
) -> list[dict]:
    """
    Identify crisis windows by:
      1) starting as soon as drawdown < 0,
      2) tracking the deepest trough,
      3) only recording the window if that trough ≤ -threshold.

    Parameters
    ----------
    drawdown_series : pd.Series
        Series of drawdown values (e.g., from calculate_cumulative_returns_and_drawdowns).
    threshold : float, default 0.10
        The drawdown threshold (in decimal form) beyond which to call a period a “crisis.”

    Returns
    -------
    List of dicts, each with keys: "Start", "Trough", "End", "Max Drawdown".
    """
    ds = drawdown_series.dropna()
    crises = []
    in_crisis = False
    start = trough = None
    max_dd = 0.0

    for date, dd_val in ds.items():
        if not in_crisis and dd_val < 0:
            in_crisis = True
            idx = ds.index.get_loc(date)
            # Peak = last non-negative drawdown date
            if idx > 0 and ds.iloc[idx-1] >= 0:
                start = ds.index[idx-1]
            else:
                start = date
            trough = date
            max_dd = dd_val

        elif in_crisis:
            if dd_val < max_dd:
                max_dd = dd_val
                trough = date
            if dd_val >= 0:
                if max_dd <= -threshold and start != trough:
                    crises.append({
                        "Start": start,
                        "Trough": trough,
                        "End": date,
                        "Max Drawdown": max_dd
                    })
                in_crisis = False

    # Close out if still in crisis at end
    if in_crisis and max_dd <= -threshold and start != trough:
        crises.append({
            "Start": start,
            "Trough": trough,
            "End": ds.index[-1],
            "Max Drawdown": max_dd
        })

    return crises



def calculate_time_to_recovery(
    cum_series: pd.Series,
    start_date: pd.Timestamp,
    trough_date: pd.Timestamp,
    unit: str = "quarterly"
) -> int|None:
    """
    Compute the # of periods (quarters or years) from trough_date until cum_series
    returns to ≥ cum_series[start_date]. Returns int or None if never recovers.
    """
    if start_date not in cum_series.index or trough_date not in cum_series.index:
        return None

    target = cum_series.loc[start_date]
    sub = cum_series.loc[trough_date:]
    rec = sub[sub >= target].index
    if len(rec) == 0:
        return None

    days = (rec[0] - trough_date).days
    if unit == "annual":
        return int(round(days / 365.25))
    elif unit == "quarterly":
        return int(round(days / 91.3125))
    else:
        raise ValueError("unit must be 'annual' or 'quarterly'")


def calculate_sei_returns(
    allocations_df: pd.DataFrame,
    returns_q: pd.DataFrame
) -> pd.Series:
    """
    Build the Synthetic Endowment Index returns series:
      SEI_t = ∑_i weight_{i,t} * return_{i,t}
    where allocations_df is the long‐form weights and returns_q has matching columns.
    """
    from utils import map_allocations_to_periods  # imported here to avoid circular
    idx_q = returns_q.index
    alloc_q = map_allocations_to_periods(allocations_df, idx_q)
    common = alloc_q.columns.intersection(returns_q.columns)
    alloc_q, returns_q = alloc_q[common], returns_q[common]
    return (alloc_q * returns_q).sum(axis=1)


def trend_sensitivity() -> list[int]:
    """
    The set of trend‐lookback speeds (in weeks) that map to your TSM columns.
    """
    return [4, 7, 12, 20, 24]


def map_allocations_to_periods(
    allocations_df: pd.DataFrame,
    date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Expand long‐form allocations_df into a DataFrame with index=date_index
    and one column per Asset Class, filling with that class’s allocation
    whenever date lies in [Start Date, End Date].
    """
    out = pd.DataFrame(index=date_index)
    for asset in allocations_df["Asset Class"].unique():
        sub = allocations_df[allocations_df["Asset Class"] == asset]
        for _, row in sub.iterrows():
            rng = pd.date_range(row["Start Date"], row["End Date"], freq=date_index.freq)
            out.loc[out.index.intersection(rng), asset] = row["Allocation"]
    return out.fillna(0.0)

###############################################################################
#                       HEDGE‑EFFICIENCY HELPERS
###############################################################################
def peak_to_trough_return(series: pd.Series,
                          start: pd.Timestamp,
                          trough: pd.Timestamp) -> float | None:
    """
    Raw return from *start* (portfolio peak) to *trough* date:

        r = (1 + R_trough) / (1 + R_start) − 1

    Parameters
    ----------
    series : pd.Series
        Periodic returns (decimal) indexed by date.
    start, trough : pd.Timestamp
        Dates defining the peak ➜ trough window.

    Returns
    -------
    float | None
        Decimal return or None if either date missing.
    """
    if start not in series.index or trough not in series.index:
        return None
    r_start, r_trough = series.loc[start], series.loc[trough]
    return (1 + r_trough) / (1 + r_start) - 1.0


def median_crisis_payoff(series: pd.Series,
                         crises: list[dict]) -> float | None:
    """
    Median peak‑to‑trough payoff of a hedge *series* across a list of
    crisis dictionaries of the form produced by `detect_crisis_periods`.

    Returns `None` if no valid windows are found.
    """
    vals = []
    for c in crises:
        ret = peak_to_trough_return(series, c["Start"], c["Trough"])
        if ret is not None:
            vals.append(ret)
    return np.nanmedian(vals) if vals else None


def protection_cost_ratio(crisis_payoff: float,
                          normal_mean: float) -> float | None:
    """
    **PCR** = average (or median) crisis payoff divided by the hedge's
    average return in normal periods.  Sign‐agnostic: a negative carry
    produces a negative denominator.

        PCR = payoff / μ_normal
    """
    return None if normal_mean in (None, 0, np.nan) else crisis_payoff / normal_mean


def cost_adjusted_risk_reduction(crisis_payoff: float,
                                 normal_mean: float) -> float | None:
    """
    **CARR** = |crisis payoff| divided by |normal‐period mean return|.

    Interpreted as: “how many percentage‑points of drawdown relief do
    I buy per percentage‑point of long‑run carry drag (or gain).”
    """
    if normal_mean in (None, 0, np.nan):
        return None
    return abs(crisis_payoff) / abs(normal_mean)
