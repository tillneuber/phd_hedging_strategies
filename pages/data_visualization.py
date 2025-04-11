import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
    Create a stacked area chart of the asset allocations over time.
    Uses a grayscale color palette and styling appropriate for publication.
    Inserts a vertical line and shaded region before 2003 to highlight backfilled data.
    """
    asset_classes = df.columns.tolist()
    
    # Grayscale palette
    base_colors = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"]
    if len(asset_classes) > len(base_colors):
        from itertools import cycle
        colors = [c for _, c in zip(range(len(asset_classes)), cycle(base_colors))]
    else:
        colors = base_colors[:len(asset_classes)]
    
    x = df.index
    y_values = [df[col].values for col in asset_classes]
    
    fig, ax = plt.subplots(figsize=(8, 4.6))
    
    # Stacked area chart
    ax.stackplot(x, *y_values, labels=asset_classes, colors=colors, edgecolor='black', linewidth=0.5)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Allocation (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0, 100, 11))

    # Insert vertical line at 2003-06-30 (or simply 2003 if you prefer)
    vertical_line_date = pd.to_datetime("2003-06-30")
    ax.axvline(x=vertical_line_date, color='black', linestyle='--', linewidth=1)
    
    # Shade the region before 2003-06-30 to indicate backfilled data
    if df.index.min() < vertical_line_date:
        ax.axvspan(df.index.min(), vertical_line_date, color='gray', alpha=0.2)
    
    # Legend at the bottom
    ax.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(asset_classes),
        frameon=False,
        fontsize=8
    )
    
    # Clean up spines
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
    
if __name__ == '__main__':
    main()
