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
    df = df * 100  # Remove this line if your CSV already has 0–100 values
    
    # Sort by date (in case the CSV is unordered)
    df.sort_index(inplace=True)
    return df

def create_area_chart(df):
    """
    Create a stacked area chart of the asset allocations over time.
    Uses a grayscale color palette and styling appropriate for publication.
    """
    # Get asset classes in the order provided by the CSV
    asset_classes = df.columns.tolist()
    
    # Define a grayscale palette (one color per asset class)
    # Lightest for the bottom area, darkest for the top.
    base_colors = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"]
    if len(asset_classes) > len(base_colors):
        # Cycle through colors if more asset classes than the palette
        from itertools import cycle
        colors = [c for _, c in zip(range(len(asset_classes)), cycle(base_colors))]
    else:
        colors = base_colors[:len(asset_classes)]
    
    # Prepare data for stackplot
    x = df.index
    y_values = [df[col].values for col in asset_classes]
    
    fig, ax = plt.subplots(figsize=(8, 4.6))
    
    # Stacked area chart
    ax.stackplot(
        x, 
        *y_values, 
        labels=asset_classes, 
        colors=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Format x-axis to show dates in "Mon Year" format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    
    ax.set_ylabel("Allocation (%)", fontsize=8)
    ax.set_ylim(0, 100)  # Ensure we see 0–100 on the y-axis
    ax.set_yticks(np.linspace(0, 100, 11))
    
    # Legend at the bottom
    ax.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(asset_classes),
        frameon=False,
        fontsize=8
    )
    
    # Remove the top and right spines for a cleaner look
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
