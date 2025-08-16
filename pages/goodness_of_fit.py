# pages/goodness_of_fit.py  ── v3  (t-stats via SciPy, JoF-style plots)
# --------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from textwrap import dedent
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress, t  
import scipy.stats as stats

# ── attempt imports for regression back-ends ─────────────────────────
try:
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

try:
    from scipy.stats import linregress
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

if not (HAVE_SM or HAVE_SCIPY):
    st.error("Install either `statsmodels` or `scipy` to obtain t-statistics.")
    st.stop()

DATA_DIR = Path(__file__).parent.parent / "data"

# ── loader helpers (unchanged) ───────────────────────────────────────
def load_allocations():
    cols = ["Year", "Public Equity", "PE/VC", "Hedge Funds",
            "Real Assets & ILBs", "Fixed Income", "Private Credit", "Cash"]
    df = pd.read_csv(DATA_DIR / "hist_endowment_saa.csv", sep=";", names=cols, header=0)
    df = df.melt(id_vars=["Year"], var_name="Asset Class", value_name="Weight")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Start"]  = pd.to_datetime(df["Year"].astype(str)) + pd.DateOffset(months=6)
    df["End"]    = df["Start"] + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    return df

def load_quarterly_returns():
    q = pd.read_csv(DATA_DIR / "quarterly_returns.csv", sep=";", header=0)
    q["Date"] = pd.to_datetime(q["Date"], format="%d.%m.%Y", errors="coerce")
    q.set_index("Date", inplace=True)
    q.index = q.index + pd.offsets.MonthEnd(0)
    for c in q.columns:
        q[c] = q[c].astype(str).str.rstrip("%").astype(float) / 100.0
    base = pd.Timestamp("1999-07-01")
    if base not in q.index:
        q.loc[base] = 0.0
    q.sort_index(inplace=True)
    return q



def load_individual_endowments():
    df = pd.read_csv(DATA_DIR / "individual_endowments.csv", sep=";", header=0)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df.set_index("Date", inplace=True)
    for c in df.columns:
        df[c] = df[c].astype(str).str.rstrip("%").astype(float) / 100.0
    base = pd.Timestamp("1999-07-01")
    if base not in df.index:
        df.loc[base] = 0.0
    df.sort_index(inplace=True)
    return df

# ── SEI construction (identical to main app) ─────────────────────────
def compute_sei_quarterly():
    alloc = load_allocations()
    qr    = load_quarterly_returns()

    start = max(alloc["Start"].min(), qr.index.min())
    end   = min(alloc["End"].max(),  qr.index.max())
    q_idx = pd.date_range(start, end, freq="Q-DEC")

    w = pd.DataFrame(index=q_idx, columns=alloc["Asset Class"].unique())
    for a in alloc["Asset Class"].unique():
        sub = alloc[alloc["Asset Class"] == a]
        for _, r in sub.iterrows():
            rng = pd.date_range(r["Start"], r["End"], freq="Q-DEC")
            w.loc[w.index.isin(rng), a] = r["Weight"]
    w = w.fillna(method="ffill")

    common = w.columns.intersection(qr.columns)
    w, r = w[common], qr[common].reindex(q_idx).fillna(0.0)
    sei_q = (w * r).sum(axis=1)
    return sei_q

def quarterly_to_fy(qret):
    cum = (1+qret).cumprod()
    fy = cum[cum.index.month == 6]      # 30 Jun fiscal year-end
    return fy.pct_change().dropna()

# ── regression metrics ───────────────────────────────────────────────
def ols_metrics(y: pd.Series, x: pd.Series) -> dict[str, float] | None:
    """
    OLS:  y = α + β·x     (returns in decimal form)

    The t-statistic for β is now computed against H₀: β = 1,
    i.e. t = (β̂ − 1) / SE(β̂).
    """
    y, x = y.align(x, join="inner")
    n = len(y)
    if n < 3:
        return None

    xv, yv = x.values, y.values
    slope, intercept, r, _, se_beta = stats.linregress(xv, yv)

    # residual variance σ̂² = RSS / (n − 2)
    y_hat  = intercept + slope * xv
    sigma2 = np.sum((yv - y_hat) ** 2) / (n - 2)

    # SE(α̂)
    sxx       = np.sum((xv - xv.mean()) ** 2)
    se_alpha  = np.sqrt(sigma2 * (1 / n + (xv.mean() ** 2) / sxx))

    # ▶ NEW: test β̂ against 1, not 0
    t_beta  = (slope - 1.0) / se_beta if se_beta > 0 else np.nan
    p_beta  = 2 * stats.t.sf(np.abs(t_beta), df=n - 2) if np.isfinite(t_beta) else np.nan

    # α test unchanged (H₀: α = 0)
    t_alpha = intercept / se_alpha if se_alpha > 0 else np.nan
    p_alpha = 2 * stats.t.sf(np.abs(t_alpha), df=n - 2) if np.isfinite(t_alpha) else np.nan

    return dict(
        alpha=intercept,
        beta=slope,
        rho=r,
        r2=r ** 2,
        rmse=np.sqrt(sigma2),
        t_beta=t_beta,
        p_beta=p_beta,
        t_alpha=t_alpha,
        p_alpha=p_alpha,
    )


def scatter_plot(x, y, series_name):
    """
    Compare annual % returns of the Synthetic Endowment Index (SEI)
    to another series (Average Endowment, Yale, Stanford, Harvard).

    Parameters
    ----------
    x : pd.Series
        SEI decimal returns.
    y : pd.Series
        Comparison series decimal returns.
    series_name : str
        Name of the comparison series; used for the y-axis label.
        If it contains 'Average', we hard-code the full NACUBO label.
    """
    # ------------------------------------------------------
    # 1)  Align and drop non-finite observations
    # ------------------------------------------------------
    aligned = x.to_frame("x").join(y.to_frame("y"), how="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()

    if aligned.empty:
        raise ValueError("No overlapping, finite data for scatter plot.")

    # ------------------------------------------------------
    # 2)  Convert to percentage space for nicer ticks
    # ------------------------------------------------------
    x_pct = aligned["x"] * 100.0
    y_pct = aligned["y"] * 100.0

    # ------------------------------------------------------
    # 3)  Ordinary-least-squares fit (slope & intercept)
    # ------------------------------------------------------
    coef = np.polyfit(x_pct, y_pct, deg=1)
    slope, intercept = coef

    # ------------------------------------------------------
    # 4)  Symmetric axis limits
    # ------------------------------------------------------
    lim = np.ceil(max(np.abs(x_pct).max(), np.abs(y_pct).max()) / 5.0) * 5.0
    lim = max(lim, 5)          # make sure we get at least ±5 %
    xmin, xmax = -lim, lim
    ymin, ymax = -lim, lim

    # ------------------------------------------------------
    # 5)  Build the figure
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    # Grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="#aaaaaa")

    # Scatter points (smaller, grey)
    ax.scatter(
        x_pct, y_pct,
        s=18, color="#555555", alpha=0.8, zorder=3
    )

    # Regression line (solid, thin)
    xx = np.linspace(xmin, xmax, 200)
    ax.plot(xx, slope * xx + intercept, color="black", linewidth=1.0, zorder=2)

    # Axes formatting
    pct_fmt = FuncFormatter(lambda v, _pos: f"{v:.0f}%")
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # Labels
    ax.set_xlabel("Synthetic Endowment Index (%)", fontsize=9)
    if "average" in series_name.lower():
        y_label = "Average Endowment (NACUBO) (%)"
    else:
        y_label = f"{series_name} (%)"
    ax.set_ylabel(y_label, fontsize=9)

    # Tick parameters (smaller ticks & labels)
    ax.tick_params(axis="both", which="both", labelsize=8, length=3)

    # Tight layout
    fig.tight_layout(pad=0.3)

    return fig

# ── Streamlit page ───────────────────────────────────────────────────
def main():
    st.title("Goodness-of-Fit:  SEI vs. Reported Endowments")

    sei_fy = quarterly_to_fy(compute_sei_quarterly())
    sei_fy.name = "SEI"

    indiv = load_individual_endowments()
    targets = {
        "Average Endowment (NACUBO)": indiv["Average Endowment (NACUBO)"],
        "Yale": indiv["Yale"],
        "Stanford": indiv["Stanford"],
        "Harvard": indiv["Harvard"],
    }

      # ── metrics table ────────────────────────────────────────────────
    rows = []
    for name, s in targets.items():
        m = ols_metrics(s, sei_fy)
        if m is None:
            continue
        rows.append({
            "Series": name,
            "ρ":          f"{m['rho']:.2f}",
            "β̂":         f"{m['beta']:.2f}",
            "t(β̂)":      f"{m['t_beta']:.2f}",
            "p-val(β̂)":  f"{m['p_beta']:.3f}",
            "α̂":         f"{m['alpha']:.1%}",
            "t(α̂)":      f"{m['t_alpha']:.2f}",
            "p-val(α̂)":  f"{m['p_alpha']:.3f}",
            "R²":         f"{m['r2']:.2f}",
            "RMSE":       f"{m['rmse']:.1%}",
        })

    st.subheader("Fiscal-Year Regressions on the Synthetic Endowment Index, 2000–2024")
    st.dataframe(pd.DataFrame(rows).set_index("Series"))

    st.caption(
        dedent("""\
        β̂, t-statistic and p-value come from an OLS of each reported series on the
        SEI (annual fiscal-year returns).  When *statsmodels* is unavailable,
        the page falls back to *scipy*’s `linregress`, which produces identical
        slope and (non-robust) t-statistics.  For annual data, HAC corrections
        rarely change inference; they can be added with
        `cov_type="HAC", cov_kwds={"maxlags":2}` in *statsmodels* if desired.""")
    )

    # ── scatter plots ────────────────────────────────────────────────
    for name, s in targets.items():
        y, x = s.align(sei_fy, join="inner")
        if len(y) < 3:
            continue
        fig = scatter_plot(x, y, name.split()[0])
        st.pyplot(fig)

if __name__ == "__main__":
    main()
