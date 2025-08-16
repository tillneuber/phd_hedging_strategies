import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


###############################################################################
# 1. Hard‑coded toy time‑series (periods t0 … t7)
###############################################################################

PUE_RET_PCT = [
    4.50, 5.00, 7.00, -8.00, -15.00, -5.00
]
SEI_RET_PCT = [
    4.50, 4.65, 5.36, 1.35, -3.56, -3.99
]

# Convert to decimals
pue_r = np.array(PUE_RET_PCT) / 100.0
sei_r = np.array(SEI_RET_PCT) / 100.0

###############################################################################
# 2. Build NAV curves (start = 100)
###############################################################################

def nav_curve(returns: np.ndarray, start_val: float = 100.0) -> np.ndarray:
    """Compound a return stream into a NAV series."""
    nav = np.empty(len(returns) + 1)
    nav[0] = start_val
    for i, r in enumerate(returns, start=1):
        nav[i] = nav[i - 1] * (1.0 + r)
    return nav

pue_nav = nav_curve(pue_r)
sei_nav = nav_curve(sei_r)

###############################################################################
# 3. One‑period look‑back TSM signals
###############################################################################
#   +1 = long / no hedge, ‑1 = short / fully hedged
#   First period (t0) is assumed *long* because no prior history.

def tsm_signal(returns: np.ndarray) -> np.ndarray:
    sig = np.ones_like(returns, dtype=int)
    sig[1:] = np.where(returns[:-1] >= 0.0, 1, -1)
    return sig

pue_sig = tsm_signal(pue_r)
sei_sig = pue_sig

###############################################################################
# 4. Build the figure  ——  NAV‐only panel, no grid, aligned style & colours
###############################################################################
def build_chart() -> plt.Figure:
    """
    Single-panel illustration for the lag-match hypothesis.
    • Shows only the NAV paths (no return panel).
    • Uses the same dashed-line style and colours as the other paper figures:
        – Hypothetical Endowment Portfolio  → black,   dashed
        – Hypothetical Public Equity Portfolio → dim-gray, dashed
    • Shades contiguous quarters in which the (one-period) TSM signal is short.
    • X-axis labelled “Quarter”.
    """
    # --- Matplotlib defaults -------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,          # no grid lines
    })

    fig, ax_nav = plt.subplots(figsize=(8, 3.0), constrained_layout=True)

    # ── Shade quarters where the SEI TSM signal is short (−1) ────────────────
    run_start = None
    for q_idx, sig in enumerate(sei_sig):
        if sig < 0 and run_start is None:
            run_start = q_idx           # first hedged quarter
        elif sig >= 0 and run_start is not None:
            ax_nav.axvspan(run_start, q_idx, color="lightgray", alpha=0.4, zorder=0)
            run_start = None
    if run_start is not None:           # open run reaching the end
        ax_nav.axvspan(run_start, len(sei_sig), color="lightgray", alpha=0.4, zorder=0)

    # ── NAV lines ------------------------------------------------------------
    endow_lbl = "Hypothetical Endowment Portfolio"
    equity_lbl = "Hypothetical Public Equity Portfolio"

    ax_nav.plot(np.arange(len(sei_nav)), sei_nav,
                linestyle="--", color="black",  linewidth=1.4,
                marker="s",  markersize=4, label=endow_lbl)
    ax_nav.plot(np.arange(len(pue_nav)), pue_nav,
                linestyle="--", color="dimgray", linewidth=1.4,
                marker="o", markersize=4, label=equity_lbl)

    # ── Axes cosmetics -------------------------------------------------------
    ax_nav.set_ylabel("NAV", fontsize=9)
    ax_nav.set_xlabel("Quarter", fontsize=9)
    ax_nav.tick_params(labelsize=8)
    ax_nav.legend(frameon=False, fontsize=8, loc="upper left")

    return fig


###############################################################################
# 5. Streamlit UI
###############################################################################

def main():
    st.title("Lag‑Match Hypothesis: Illustration")

    fig = build_chart()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Underlying numbers"):
        st.write("### Period returns (%)")
        st.dataframe(
            {
                "Period": np.arange(len(pue_r)),
                "Public Equity": PUE_RET_PCT,
                "SEI": SEI_RET_PCT,
                "PuE TSM signal": pue_sig,
                "SEI TSM signal": sei_sig,
            },
            hide_index=True,
        )
        st.write("### NAV paths")
        st.dataframe(
            {
                "Period": np.arange(len(pue_nav)),
                "PuE NAV": pue_nav,
                "SEI NAV": sei_nav,
            },
            hide_index=True,
        )


if __name__ == "__main__":
    main()
