import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

###############################################################################
#             ASSET-CLASS CUMULATIVE-RETURN & DRAWDOWN UTILITIES
###############################################################################
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from streamlit_app import load_and_preprocess_data, calculate_cumulative_and_dd


def load_asset_class_returns(freq: str = "Q") -> pd.DataFrame:
    """
    Pull periodic total-return series for the endowment’s strategic
    asset-class buckets (Public Equity, Fixed Income, Real Assets & ILBs,
    Private Credit, PE/VC, Hedge Funds, Cash …).

    • We first call `load_and_preprocess_data` (same helper used elsewhere)  
      to obtain the master return matrix `ret_full`.  
    • We intersect its columns with the allocation columns returned by
      `load_allocations()` so that only genuine asset-class series remain.
    • The result is resampled to *quarter-end* observations by default
      (`freq="Q"`), picking the last available monthly datapoint in each
      quarter.

    Returns
    -------
    pd.DataFrame
        Index  : period-end dates  
        Columns: one per asset class, values are total-return levels
    """
    # 1) Master return matrix + allocation column names
    _, ret_full = load_and_preprocess_data()
    alloc_cols  = load_allocations().columns.tolist()

    asset_cols  = [c for c in ret_full.columns if c in alloc_cols]
    if not asset_cols:
        raise ValueError("No overlapping asset-class columns between "
                         "`ret_full` and `alloc_cols`.")

    # 2) Quarter-end (default) or leave at original frequency
    if freq.upper().startswith("Q"):
        ret_asset = (
            ret_full[asset_cols]
            .resample("Q")  # period-end
            .last()
            .dropna(how="all")
        )
    else:
        ret_asset = ret_full[asset_cols].dropna(how="all")

    return ret_asset


def build_asset_class_cum_and_dd(ret_df: pd.DataFrame):
    """
    Compute cumulative-return indices **and** percentage drawdowns for each
    column in `ret_df`.

    Returns
    -------
    cum_dict : dict[str, pd.Series]
        {asset_class: cumulative_index_series}
    dd_dict  : dict[str, pd.Series]
        {asset_class: drawdown_series}
    """
    cum_dict, dd_dict = {}, {}
    for col in ret_df.columns:
        series = ret_df[col].dropna()
        if series.empty:
            continue
        cum, dd = calculate_cumulative_and_dd(series)
        cum_dict[col] = cum
        dd_dict[col]  = dd
    return cum_dict, dd_dict


def create_asset_class_drawdown_chart(
    cum_dict: dict,
    dd_dict : dict,
    asset_keys: list[str] | None = None,
    title: str = "Asset-Class Cumulative Returns and Drawdowns"
):
    """
    Two-panel figure for every strategic asset class,
    **now in colour (Tab10 cycle)** instead of greyscale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    from itertools import cycle

    if asset_keys is None:
        asset_keys = list(cum_dict.keys())

    # --- colour cycle identical to other line charts -----------------------
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # Tab10 default
    colors      = [c for _, c in zip(range(len(asset_keys)), cycle(color_cycle))]
    colour_map  = dict(zip(asset_keys, colors))

    # --- figure skeleton ---------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"hspace": 0.05},  # space between panels
        constrained_layout=True        # use CL; drop tight_layout
    )

    # (1) cumulative returns
    for k in asset_keys:
        ser = cum_dict.get(k, pd.Series(dtype=float)).dropna()
        if ser.empty:
            continue
        ax1.plot(ser.index, ser.values, label=k,
                 color=colour_map[k], linewidth=1.3)
    ax1.set_ylabel("Cumulative\nReturn", fontsize=8)
    ax1.tick_params(labelsize=8)

    # (2) drawdowns
    for k in asset_keys:
        ser = dd_dict.get(k, pd.Series(dtype=float)).dropna()
        if ser.empty:
            continue
        ax2.plot(ser.index, ser.values, label=k,
                 color=colour_map[k], linewidth=1.3)
    ax2.set_ylabel("Drawdown", fontsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.tick_params(labelsize=8)

    # --- x-axis & legend ----------------------------------------------------
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=3,
               frameon=False, fontsize=8)

    fig.suptitle(title, y=1.02, fontsize=10)
    plt.subplots_adjust(bottom=0.18)   # leave room for legend
    return fig


def load_allocations():
    """
    Load the historical endowment SAA CSV.
    Assumes columns: Year, Public Equity, PE/VC, Hedge Funds,
    Real Assets & ILBs, Fixed Income, Private Credit, Cash.
    Converts the Year into a date (set at June 30 of each year)
    and returns a DataFrame indexed by date.
    """
    alloc_cols = [
        'Year', 'Public Equity', 'PE/VC', 'Hedge Funds',
        'Real Assets & ILBs', 'Fixed Income', 'Private Credit', 'Cash'
    ]
    df = pd.read_csv('data/hist_endowment_saa.csv', sep=';', names=alloc_cols, header=0)
    
    # Convert "Year" to a date: using June 30 as the representative date
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-06-30', errors='coerce')
    df.set_index('Date', inplace=True)
    df.drop(columns=['Year'], inplace=True)
    
    # Convert to numeric; if data is in decimals (0.20 for 20%), multiply by 100
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df * 100  # Remove if CSV already has % in 0–100

    # Sort by date (in case the CSV is unordered)
    df.sort_index(inplace=True)
    return df

def create_area_chart(df):
    """
    Stacked area chart of the endowment’s SAA.
    • Illiquid assets (PE/VC, Private Credit, Real Assets & ILBs) sit at the
      bottom of the stack.
    • Those three layers are over-printed with a hatch pattern so they pop
      out even in greyscale or photocopies.
    """
    # ----- 1) economic ordering ------------------------------------------- #
    illiquid = ["PE/VC", "Private Credit", "Real Assets & ILBs"]
    liquid   = ["Public Equity", "Fixed Income", "Hedge Funds", "Cash"]
    ordered_cols = [c for c in illiquid + liquid if c in df.columns]
    df = df[ordered_cols]

    # ----- 2) greys (dark → light) ---------------------------------------- #
    palette = ["#252525", "#525252", "#737373",   # illiquid
               "#969696", "#bdbdbd", "#d9d9d9", "#f0f0f0"]  # liquid
    from itertools import cycle, islice
    colors = list(islice(cycle(palette), len(ordered_cols)))

    # ----- 3) plot & add hatching ----------------------------------------- #
    x = df.index
    y = [df[c].values for c in ordered_cols]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    polys = ax.stackplot(x, *y, labels=ordered_cols,
                         colors=colors, edgecolor='black', linewidth=0.5)

    # `polys` is bottom-to-top → same order as `ordered_cols`
    hatch_map = {"PE/VC": "////", "Private Credit": "xxxx",
                 "Real Assets & ILBs": "\\\\\\"}      # pick any patterns
    for poly, col in zip(polys, ordered_cols):
        if col in hatch_map:
            poly.set_hatch(hatch_map[col])

    # ----- 4) cosmetics unchanged ---------------------------------------- #
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Allocation (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0, 100, 11))

    vline = pd.to_datetime("2003-06-30")
    ax.axvline(vline, color='black', linestyle='--', linewidth=1)
    if df.index.min() < vline:
        ax.axvspan(df.index.min(), vline, color='gray', alpha=0.2)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
              ncol=len(ordered_cols), frameon=False, fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.2)
    return fig

def main():
    st.title("Data Visualizations")
    st.subheader("Evolution of Strategic Asset Allocation")
    
    # Load the historical asset allocation data
    allocations = load_allocations()
    
    # Create and display the stacked area chart
    fig = create_area_chart(allocations)
    st.pyplot(fig)

    # -------------------------------------------------------------------------
    #  Drawdowns for all strategic asset classes
    # -------------------------------------------------------------------------
    st.subheader("Asset-Class Cumulative Returns & Drawdowns")

    ret_assets  = load_asset_class_returns()             # quarterly returns
    cum_dict, dd_dict = build_asset_class_cum_and_dd(ret_assets)

    if cum_dict:                                         # at least one series
        fig_draw = create_asset_class_drawdown_chart(cum_dict, dd_dict)
        st.pyplot(fig_draw)
    else:
        st.info("No asset-class return series available.")


    # -------------------------------------------------------------------------
    #  Focus chart: Public Equity  vs.  PE/VC
    # -------------------------------------------------------------------------
    st.subheader("Public Equity vs. PE/VC – Cumulative Returns & Drawdowns")

    focus_keys = [k for k in ["Public Equity", "PE/VC"] if k in cum_dict]

    if len(focus_keys) == 2:       # both series exist
        fig_focus = create_asset_class_drawdown_chart(
            cum_dict,
            dd_dict,
            asset_keys=focus_keys,
            title="Public Equity vs. PE/VC"
        )
        st.pyplot(fig_focus)
    else:
        st.info("Both 'Public Equity' and 'PE/VC' series are required for this chart.")

    # -------------------------------------------------------------------------
    #  Focus chart – Public Equity vs. PE/VC  (GFC crisis window only)
    # -------------------------------------------------------------------------
    st.subheader("Public Equity vs. PE/VC – GFC Crisis Window")

    import pandas as pd

    # --- 1) hard-coded crisis dates -------------------------------------------
    gfc_start = pd.Timestamp("2007-12-31")   # SEI peak
    gfc_end   = pd.Timestamp("2010-12-31")   # SEI full recovery
    # (change these two dates if your paper specifies others)

    # --- 2) extract the two series & slice to the window ----------------------
    focus_keys = ["Public Equity", "PE/VC"]

    cum_gfc = {
        k: v.loc[gfc_start:gfc_end]
        for k, v in cum_dict.items()
        if k in focus_keys
    }
    dd_gfc = {
        k: v.loc[gfc_start:gfc_end]
        for k, v in dd_dict.items()
        if k in focus_keys
    }

    # --- 3) plot --------------------------------------------------------------
    if all(len(s) > 1 for s in cum_gfc.values()):
        fig_focus_gfc = create_asset_class_drawdown_chart(
            cum_gfc, dd_gfc, asset_keys=focus_keys,
            title=f"Public Equity vs. PE/VC – GFC "
                f"({gfc_start.date()} to {gfc_end.date()})"
        )
        st.pyplot(fig_focus_gfc)
    else:
        st.info("Both series must be available for the specified GFC window.")
        
if __name__ == '__main__':
    main()
