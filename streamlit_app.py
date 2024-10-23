import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, timezone
import urllib.parse
from windrose import WindroseAxes
import matplotlib.pyplot as plt

# Define API credentials and endpoint
API_BASE_URL = "https://db.oceans.xyz/api/v2/tables/md_pmwumtlegitpyy/records"
API_TOKEN = "xR1XCcl-7fgHi7RcWFjtOstThQ8noxHfxtKcll_A"

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_live_data(start_date: str, end_date: str):
    headers = {
        "xc-token": API_TOKEN,
        "Content-Type": "application/json"
    }

    where_clause = f"((CreatedAt,ge,exactDate,{start_date})~and(CreatedAt,le,exactDate,{end_date}))"
    
    # First make a request to get total count
    params = {
        "where": where_clause,
        "limit": 1
    }
    
    response = requests.get(API_BASE_URL, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"Failed to fetch data count: {response.status_code}")
        return pd.DataFrame(), {}
        
    total_records = response.json().get("pageInfo", {}).get("totalRows", 0)
    
    # Initialize variables for pagination
    offset = 0
    limit = 1000
    all_data = []
    
    # Calculate number of iterations needed
    total_iterations = (total_records + limit - 1) // limit
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(total_iterations):
        # Update progress
        progress = (iteration + 1) / total_iterations
        progress_bar.progress(progress)
        status_text.text(f"Fetching data... {int(progress * 100)}%")
        
        params = {
            "offset": offset,
            "limit": limit,
            "where": where_clause,
        }
        
        response = requests.get(API_BASE_URL, headers=headers, params=params)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch data batch: {response.status_code}")
            break
            
        batch_data = response.json().get("list", [])
        
        if not batch_data:
            break
            
        all_data.extend(batch_data)
        offset += limit
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_data), {"totalRows": len(all_data)}

def align_comparison_data(current_data, comparison_data):
    """
    Aligns comparison data to match the current year's time window
    """
    if comparison_data.empty or current_data.empty:
        return comparison_data
        
    # Get the current year's data timespan
    current_year = datetime.now().year
    comparison_year = comparison_data['timestamp'].dt.year.iloc[0]
    
    # Calculate the year difference
    year_diff = current_year - comparison_year
    
    # Shift the comparison data to align with current year
    comparison_data['timestamp'] = comparison_data['timestamp'] + pd.DateOffset(years=year_diff)
    
    # Filter comparison data to match current data's date range
    start_date = current_data['timestamp'].min()
    end_date = current_data['timestamp'].max()
    mask = (comparison_data['timestamp'] >= start_date) & (comparison_data['timestamp'] <= end_date)
    
    return comparison_data[mask]

# Streamlit application
st.title("Live Wind Data Visualization")

# Sidebar options
st.sidebar.header("Select Time Range")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
end_date = st.sidebar.date_input("End Date", datetime.now()).strftime('%Y-%m-%d')

# Dropdown to select additional year for comparison
year_option = st.sidebar.selectbox("Select Additional Year for Comparison", options=[None, 2022, 2023, 2024], index=0)

# Add a warning for large date ranges
date_difference = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
if date_difference > 90:
    st.sidebar.warning(f"You've selected a {date_difference} day range. Large date ranges may take longer to load.")

# Validate that start_date is before end_date
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
else:
    # Fetch data based on the selected time range
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            data, page_info = fetch_live_data(start_date, end_date)
            comparison_data = pd.DataFrame()
            if year_option:
                start_date_comparison = start_date.replace(str(datetime.now().year), str(year_option))
                end_date_comparison = end_date.replace(str(datetime.now().year), str(year_option))
                comparison_data, _ = fetch_live_data(start_date_comparison, end_date_comparison)
    else:
        # Fetch the last month of records by default
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        with st.spinner("Fetching data..."):
            data, page_info = fetch_live_data(start_date, end_date)
            comparison_data = pd.DataFrame()

    # Check if data is fetched successfully
    if not data.empty:
        # Rename columns to match the original dataset attributes
        data.rename(columns={
            'CreatedAt': 'timestamp',
            'AirWindSpeedAvg': 'wind_speed',
            'AirWindDirectionAvg': 'wind_direction'
        }, inplace=True)

        # Convert timestamp to datetime and sort by timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S%z')
        data = data.sort_values('timestamp')

        # Display the date range and number of records
        st.write(f"Showing {len(data):,} records from {data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}")

        # Ensure required columns are present
        if 'wind_speed' in data.columns and 'wind_direction' in data.columns:
            # Line plot for wind speed over time
            st.subheader("Wind Speed Over Time")
            fig = px.line(data, x='timestamp', y='wind_speed', 
                         title='Wind Speed Over Time', 
                         labels={'timestamp': 'Date', 'wind_speed': 'Wind Speed (m/s)'})

            # If comparison data is available, add it to the plot
            if not comparison_data.empty:
                comparison_data.rename(columns={
                    'CreatedAt': 'timestamp',
                    'AirWindSpeedAvg': 'wind_speed',
                    'AirWindDirectionAvg': 'wind_direction'
                }, inplace=True)
                comparison_data['timestamp'] = pd.to_datetime(comparison_data['timestamp'], format='%Y-%m-%d %H:%M:%S%z')
                comparison_data = comparison_data.sort_values('timestamp')
                
                # Align comparison data with current year's window
                comparison_data = align_comparison_data(data, comparison_data)
                
                # Add comparison data to plot
                fig.add_scatter(x=comparison_data['timestamp'], 
                              y=comparison_data['wind_speed'], 
                              mode='lines', 
                              name=f'Wind Speed {year_option}')

            st.plotly_chart(fig)

            # Wind rose chart
            st.subheader("Wind Rose of Wind Direction and Speed")
            ws = data['wind_speed'].to_numpy()
            wd = data['wind_direction'].to_numpy()

            plt.figure(figsize=(10, 10))
            ax = WindroseAxes.from_ax()
            binsrange = [0, 1.4, 1.8, 2.3, 3.8, 5.4, 6.9, 8.4, 10]
            ax.bar(wd, ws, normed=True, bins=binsrange, nsector=36, opening=0.9, edgecolor='white')
            ax.set_legend(bbox_to_anchor=(1.1, 1.05), title="Wind Speed (m/s)")
            plt.title("Wind Rose for Wind Direction and Speed")
            st.pyplot(plt.gcf())

            #[Previous code remains the same until the histogram section, replacing from "# Histograms" onwards:]

# Histograms
st.subheader("Comparative Histograms")

# =====================
# Wind Direction Histograms
# =====================
st.write("### Wind Direction Distribution")
col1, col2 = st.columns(2)

# Define bins for Wind Direction (0-360 degrees)
wd_bins = np.linspace(0, 360, 37)  # 36 bins

# Calculate histogram counts for current period
wd_counts, _ = np.histogram(wd, bins=wd_bins)

# Initialize max count with current period counts
max_wd_count = wd_counts.max()

# If comparison data exists, calculate its histogram counts and update max count
if not comparison_data.empty:
    comp_wd_counts, _ = np.histogram(comparison_data['wind_direction'], bins=wd_bins)
    max_wd_count = max(max_wd_count, comp_wd_counts.max())

with col1:
    plt.figure(figsize=(8, 6))
    plt.hist(wd, bins=wd_bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Wind Direction (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Wind Direction\n{datetime.now().year}')
    plt.xlim(0, 360)
    plt.ylim(0, max_wd_count * 1.05)  # Add a 5% margin on top
    st.pyplot(plt.gcf())
    plt.close()

with col2:
    if not comparison_data.empty:
        plt.figure(figsize=(8, 6))
        plt.hist(comparison_data['wind_direction'], bins=wd_bins, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Wind Direction (degrees)')
        plt.ylabel('Frequency')
        plt.title(f'Wind Direction\n{year_option}')
        plt.xlim(0, 360)
        plt.ylim(0, max_wd_count * 1.05)  # Use the same y-axis limit
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.write("No comparison data available")

# Add basic statistics for wind direction
st.write("Wind Direction Statistics:")
stats_col1, stats_col2 = st.columns(2)
with stats_col1:
    st.write(f"Current Period ({datetime.now().year}):")
    st.write(f"- Mean: {np.nanmean(wd):.1f}°")
    st.write(f"- Median: {np.nanmedian(wd):.1f}°")
    st.write(f"- Std Dev: {np.nanstd(wd):.1f}°")

with stats_col2:
    if not comparison_data.empty:
        comp_wd = comparison_data['wind_direction'].to_numpy()
        st.write(f"Comparison Period ({year_option}):")
        st.write(f"- Mean: {np.nanmean(comp_wd):.1f}°")
        st.write(f"- Median: {np.nanmedian(comp_wd):.1f}°")
        st.write(f"- Std Dev: {np.nanstd(comp_wd):.1f}°")

# =====================
# Wind Speed Histograms
# =====================
st.write("### Wind Speed Distribution")
col3, col4 = st.columns(2)

# Define bins for Wind Speed
ws_bins = 30  # Number of bins

# Determine the combined range for Wind Speed
if not comparison_data.empty:
    combined_ws = np.concatenate([ws, comparison_data['wind_speed'].dropna()])
    ws_min = combined_ws.min()
    ws_max = combined_ws.max()
else:
    ws_min = ws.min()
    ws_max = ws.max()

# Create bins based on the combined range
ws_bins_edges = np.linspace(ws_min, ws_max, ws_bins + 1)

# Calculate histogram counts for current period
ws_counts, _ = np.histogram(ws, bins=ws_bins_edges)
max_ws_count = ws_counts.max()

# If comparison data exists, calculate its histogram counts and update max count
if not comparison_data.empty:
    comp_ws_counts, _ = np.histogram(comparison_data['wind_speed'], bins=ws_bins_edges)
    max_ws_count = max(max_ws_count, comp_ws_counts.max())

with col3:
    plt.figure(figsize=(8, 6))
    plt.hist(ws, bins=ws_bins_edges, edgecolor='black', alpha=0.7)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.title(f'Wind Speed\n{datetime.now().year}')
    plt.xlim(ws_min, ws_max)
    plt.ylim(0, max_ws_count * 1.05)  # Add a 5% margin on top
    st.pyplot(plt.gcf())
    plt.close()

with col4:
    if not comparison_data.empty:
        plt.figure(figsize=(8, 6))
        plt.hist(comparison_data['wind_speed'], bins=ws_bins_edges, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Frequency')
        plt.title(f'Wind Speed\n{year_option}')
        plt.xlim(ws_min, ws_max)
        plt.ylim(0, max_ws_count * 1.05)  # Use the same y-axis limit
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.write("No comparison data available")

# Add basic statistics for wind speed
st.write("Wind Speed Statistics:")
stats_col3, stats_col4 = st.columns(2)
with stats_col3:
    st.write(f"Current Period ({datetime.now().year}):")
    st.write(f"- Mean: {np.nanmean(ws):.2f} m/s")
    st.write(f"- Median: {np.nanmedian(ws):.2f} m/s")
    st.write(f"- Std Dev: {np.nanstd(ws):.2f} m/s")

with stats_col4:
    if not comparison_data.empty:
        comp_ws = comparison_data['wind_speed'].to_numpy()
        st.write(f"Comparison Period ({year_option}):")
        st.write(f"- Mean: {np.nanmean(comp_ws):.2f} m/s")
        st.write(f"- Median: {np.nanmedian(comp_ws):.2f} m/s")
        st.write(f"- Std Dev: {np.nanstd(comp_ws):.2f} m/s")

# =====================
# Comparative Wind Roses
# =====================
if not comparison_data.empty:
    st.subheader("Comparative Wind Roses")
    col5, col6 = st.columns(2)
    
    with col5:
        plt.figure(figsize=(8, 8))
        ax = WindroseAxes.from_ax()
        ax.bar(wd, ws, bins=binsrange, nsector=36, opening=0.9, edgecolor='white', alpha=0.7)
        ax.set_legend(bbox_to_anchor=(1.1, 1.05), title="Wind Speed (m/s)")
        plt.title(f"Wind Rose\n{datetime.now().year}")
        st.pyplot(plt.gcf())
        plt.close()
    
    with col6:
        plt.figure(figsize=(8, 8))
        ax = WindroseAxes.from_ax()
        ax.bar(comparison_data['wind_direction'], comparison_data['wind_speed'], 
               bins=binsrange, nsector=36, opening=0.9, edgecolor='white', alpha=0.7, color='orange')
        ax.set_legend(bbox_to_anchor=(1.1, 1.05), title="Wind Speed (m/s)")
        plt.title(f"Wind Rose\n{year_option}")
        st.pyplot(plt.gcf())
        plt.close()
else:
    st.write("Required data columns ('wind_speed', 'wind_direction') are not available in the dataset.")
   # else:
    #    st.write("No data available for the selected time range.")