# app.py

"""
Stock Price Dashboard (Streamlit)

A compact, resume-ready dashboard for exploring equities with:
- Price + Moving Averages (20/50/200)
- Relative performance (indexed to 100 or % return since start)
- Rolling volatility (annualized)
- Drawdowns (peak-to-trough loss)

Data source: Yahoo Finance via yfinance (adjusted close)

Run:
    Option A:
        make run
    Option B:
        python3 -m venv .venv && source .venv/bin/activate
        pip install -r requirements.txt
        streamlit run app.py
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ---- App/page setup ----------------------------------------------------------

st.set_page_config(page_title="Stock Price Dashboard", layout="wide")
st.title("Stock Price Dashboard")

# ---- Sidebar controls --------------------------------------------------------
st.sidebar.header("Controls")

# Preset groups to make ticker selection quick
PRESETS = {
    "Canadian Banks (.TO)": ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"],   # RBC, TD, Scotiabank, BMO, CIBC, National Bank 
    "US Large Caps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],                 # Apple, Microsoft, Nvidia, Amazon, Google, Meta
    "ETFs/Index": ["SPY", "QQQ", "^GSPTSE", "^GSPC"]
}

# sidebar dropdown options are the keys of PRESETS, index=0 preselects first key, selected label (key) is returned & stored in preset_key
preset_key = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0) # "Canadian Banks (.TO)"

# Use the selected preset to pre-populate the multiselect
default_tickers = PRESETS[preset_key] # ["RY.TO","TD.TO","BNS.TO"]

# Sidebar Multiselect for tickers
tickers = st.sidebar.multiselect(
    "Tickers",                                                  # Label above widget
    options=sorted(set(sum(PRESETS.values(), []))),             # All unique labels, sorted A -> Z
    default=default_tickers[:3] if default_tickers else [],     # Preselect up to 3 from the chosen preset
    help="Pick 1+ symbols to load"                              # Tooltip text (hover the ?)
)

# Date range pickers; default to last 365 days (inclusive)
today = dt.date.today()
start = st.sidebar.date_input("Start date", today - dt.timedelta(days=365))
end = st.sidebar.date_input("End date", today)

# Guard against an invalid range 
if start > end:
    st.sidebar.error("Start date must be before end date.")

# Sidebar multi-select for which moving-average windows (in trading days) to plot.
# Defaults to 20/50/200; returns a list of selected window lengths (e.g., [20, 50, 200]).
# If the user deselects all options, no MA lines will be added to the chart.
ma_windows = st.sidebar.multiselect("Moving Averages (days)", [20, 50, 200], default=[20, 50, 200])

# ---- Data helpers ------------------------------------------------------------

# Cache the function's output for 1 hour (3600s) based on its inputs.
# This avoids re-downloading the same ticker/date data from Yahoo repeatedly.
# `show_spinner=False` hides the Streamlit "running" spinner when a cached
# result is served. The cache is invalidated if code or inputs change.
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers_list, start_date, end_date):
    """
    Fetch adjusted close prices for one or more tickers between start_date and end_date.

    Parameters
        tickers_list : list[str]
            Symbols to download (e.g., ["AAPL", "MSFT"])
        start_date, end_date : date-like
            Bounds for the historical download (inclusive/exclusive behavior handled by yfinance)

    Returns
        pandas.DataFrame
            Index = trading dates
            Columns = ticker symbols
            Values = adjusted close prices (float)
            Shape is always a DataFrame (even for a single ticker).
    """
    # If the user hasn't selected any tickers, return an empty frame early.
    if not tickers_list:
        return pd.DataFrame()

    # Download OHLCV data from Yahoo Finance via yfinance.
    # auto_adjust=True applies split/dividend adjustments and moves "Adj Close" into "Close".
    # progress=False disables yfinance's progress bar (cleaner in Streamlit).
    df = yf.download(
        tickers=tickers_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # If the request failed or returned an unexpected shape, bail out.
    # With multiple tickers, yfinance returns a DataFrame with top-level columns
    # like ["Open","High","Low","Close","Volume"] (and second level = tickers).
    # Only proceed if "Close" exists.
    if "Close" not in df:
        return pd.DataFrame()

    # Select only the adjusted closing prices.
    close = df["Close"].copy()

    # For a single ticker, df["Close"] is a Series. Normalize to a one-column DataFrame
    # so the function always returns a consistent shape (columns = tickers).
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers_list[0])

    # Drop columns that are entirely missing (e.g., invalid or delisted tickers).
    close = close.dropna(how="all")

    return close


def log_returns(price_df):
    """
    Compute log (continuously compounded) returns from a price series/table.

    Parameters
        price_df : pandas.Series or pandas.DataFrame
            Indexed by date (or time). Values should be strictly positive prices
            (e.g., adjusted close).

    Return
        pandas.Series or pandas.DataFrame
            Same shape as input, containing log returns:
                r_t = ln( P_t / P_{t-1} )
            The first row will be NaN because there is no previous price to compare to.
    """
    # Align with the previous observation, compute the gross return ratio P_t / P_{t-1},
    # then take the natural log to obtain the log return.
    return np.log(price_df / price_df.shift(1))


# ---- Data load & validation ---------------------------------------------------
# If the user hasn't selected any tickers yet, show a friendly hint
# and stop executing the rest of the script for this run.
if not tickers:
    st.info("Select at least one ticker in the sidebar to begin.")
    st.stop()

# Fetch adjusted-close prices for the selected tickers and date range.
prices = load_prices(tickers, start, end)

# If the download returned nothing (bad tickers, no overlap in dates, API hiccup),
# warn the user and stop to avoid downstream errors.
if prices.empty:
    st.warning("No price data returned. Check tickers/date range.")
    st.stop()

# Let the user choose a "primary" ticker for single-series visuals (MAs, KPIs, returns table).
display_ticker = st.selectbox("Primary ticker for indicators/MA chart", tickers, index=0)

# It's possible a selected ticker had no valid data and was dropped in load_prices().
# Guard against that by ensuring the chosen display_ticker exists in the prices columns.
if display_ticker not in prices.columns:
    st.warning(f"No data for {display_ticker} in the selected range.")
    st.stop()


# ---- KPIs (basic daily stats for the primary asset) --------------------------
# Compute daily log returns for the selected "primary" ticker only.
# prices[[display_ticker]] keeps it as a DataFrame (not a Series) for consistency.
rets = log_returns(prices[[display_ticker]]).dropna()

# Basic daily statistics:
# - daily_mean: average daily log return
# - daily_vol:  standard deviation of daily log returns (a measure of daily volatility)
daily_mean = rets[display_ticker].mean()
daily_vol  = rets[display_ticker].std()

# Lay out three KPI cards side-by-side.
k1, k2, k3 = st.columns(3)

# Show the average daily return as a percentage (e.g., 0.00123 -> 0.123%).
k1.metric("Daily mean return", f"{daily_mean*100:.3f}%")

# Show daily volatility (std dev of daily log returns) as a percentage.
k2.metric("Daily volatility", f"{daily_vol*100:.3f}%")

# Show how many observations (trading days) are used in these stats.
k3.metric("Observations", f"{len(rets):,}")

# Visual separator before the next section.
st.divider()


# ---- Price + Moving Averages (Plotly, $-formatted axis) ----------------------
# Section header showing which ticker's price we’re plotting.
st.subheader(f"Price & Moving Averages — {display_ticker}")

# Pull the selected ticker’s adjusted close series and ensure no missing rows.
# Rename to "Close" for a clean column label in the chart legend.
price_series = prices[display_ticker].dropna().rename("Close")

# Start the plotting DataFrame with the raw close prices.
plot_df = pd.DataFrame({"Date": price_series.index, "Close": price_series})

# For each selected moving-average window (e.g., 20/50/200),
# compute the rolling mean over that many *rows* (trading days) and
# add it as a new column (e.g., "MA20", "MA50", "MA200").
for w in ma_windows:
    plot_df[f"MA{w}"] = price_series.rolling(w).mean()

# Reshape from wide → long so Plotly can draw one line per series.
# Keeps "Date" as the identifier and stacks the other columns into:
#   - "Series": the original column name (e.g., "Close", "MA20", "MA50")
#   - "Price":  the numeric value for that date/series
# Example: [Date | Close | MA20] → rows like (Date, "Close", v1), (Date, "MA20", v2)
plot_long = plot_df.melt(id_vars="Date", var_name="Series", value_name="Price")

# Control legend and render order. List MAs first and put "Close" last so:
# - legend shows MAs first (grouped)
# - Close is plotted on top of MAs (easier to see)
cat_order = [f"MA{w}" for w in sorted(ma_windows)] + ["Close"]

# Choose colors: soft colors for MAs, thicker primary price line
color_map = {"Close": "#1f77b4", "MA20": "#9ecae1", "MA50": "#f7b6d2", "MA200": "#ff9896"}

# Create line chart:
# - color by "Series" (Close/MA20/MA50/MA200)
# - enforce category/legend order
# - map chosen colors
fig = px.line(
    plot_long, x="Date", y="Price", color="Series",
    category_orders={"Series": cat_order},
    color_discrete_map=color_map,
    title=f"Price & Moving Averages — {display_ticker}",
    labels={"Price": "Price", "Date": "Date"}
)

# Format the Y axis as currency:
# - tickprefix="$" prepends a $ to each tick label
# - separatethousands=True renders 15000 as 15,000
fig.update_yaxes(tickprefix="$", separatethousands=True)

# Global figure styling and layout tweaks
fig.update_layout(
    template="plotly_white",   # clean, minimal background + grid
    hovermode="x unified",     # one shared hover tooltip for all series at a given x (Date)
    legend_title_text="",      # remove legend title (keeps UI uncluttered)
    legend_orientation="h",    # horizontal legend row
    legend_y=1.05, legend_x=0, # position legend just above the plotting area (y>1 is above)
    margin=dict(               # outer margins in pixels (top, right, bottom, left)
        t=60, r=20, b=40, l=60
    )
)

# Iterate over each Plotly trace (one per series in the chart).
# Style primary price line ("Close") to be thicker/solid,
# and style all other lines (the moving averages) thinner and dashed
# so the Close series visually stands out.
for tr in fig.data:
    if tr.name == "Close":
        # Emphasize the actual price series
        tr.update(line=dict(width=3))               # thicker solid line
    else:
        # De-emphasize supporting series (MAs)
        tr.update(line=dict(width=2, dash="dash"))  # thinner, dashed line

# Render the Plotly figure inside Streamlit and let it expand to the container width.
st.plotly_chart(fig, use_container_width=True)


# ---- Relative performance (indexed or return %) ------------------------------
# Section header stating we are showing the Relative Performance or Return Percentage of the chosen tickers.
st.subheader("Relative performance")

# UI toggle for chart display mode.
# Shows two radio options side-by-side and returns the selected label string
# ("Indexed (100 = start)" or "Return (%)") for downstream if/else logic.
mode = st.radio("Display as", ["Indexed (100 = start)", "Return (%)"], horizontal=True)

# Require overlapping dates across tickers so they share the same baseline
aligned = prices.dropna().copy()

# If we have any overlapping data across all tickers, build the chart;
# otherwise show a helpful caption (see the else branch).
if not aligned.empty:

    # Choose what to plot based on the radio toggle:
    # - "Indexed (100 = start)": rebase each series so the first common date = 100
    # - "Return (%)": compute cumulative return since the first common date
    if mode == "Indexed (100 = start)":
        plot_vals = aligned / aligned.iloc[0] * 100   # e.g., 120 means +20% since start
        y_col = "Index"
        yaxis_title = "Index (100 = start date)"
        y_tickformat = None                           # e.g., set to ".0f" if you want whole numbers
        y_ticksuffix = ""                             # no suffix for an index
    else:  # "Return (%)"
        plot_vals = aligned / aligned.iloc[0] - 1     # keep as decimals; Plotly will format as %
        y_col = "Return"
        yaxis_title = "Return since start"
        y_tickformat = ".0%"                          # 0.12 -> "12%"
        y_ticksuffix = None

    # Convert from wide -> long so Plotly draws one trace per ticker.
    # Resulting columns: Date | Ticker | (Index or Return)
    plot_df = plot_vals.copy()
    plot_df["Date"] = plot_vals.index
    long_df = plot_df.melt(id_vars="Date", var_name="Ticker", value_name=y_col)

    # Build the line chart with a clean look and a unified hover tooltip.
    fig = px.line(
        long_df, x="Date", y=y_col, color="Ticker",
        title="Relative performance (baseline = first common date)",
        labels={"Date": "", y_col: yaxis_title}
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",     # one tooltip showing all series at the cursor's date
        legend_title_text="",
        legend_orientation="h",    # horizontal legend above the plot
        legend_y=1.05, legend_x=0,
        margin=dict(t=60, r=20, b=40, l=60)
    )

    # Apply y-axis formatting depending on the chosen mode.
    if y_tickformat:
        fig.update_yaxes(tickformat=y_tickformat)  # e.g., percent format
    if y_ticksuffix:
        fig.update_yaxes(ticksuffix=y_ticksuffix)  # optional suffix like " pts"

    # Render the Plotly figure in Streamlit and let it expand to fill the width.
    st.plotly_chart(fig, use_container_width=True)
else:
    # No common overlap across tickers → we can’t define a shared baseline.
    st.caption("Not enough overlapping data across tickers to compute a common baseline.")


# ---- Rolling volatility (annualized) -----------------------------------------
# Section header for the rolling volatility chart of the selected ticker.
st.subheader(f"Rolling volatility — {display_ticker}")

# Let the user choose the rolling window length in trading days.
# Shorter windows react faster but are noisier; longer windows are smoother but slower.
win = st.slider("Window (days)", min_value=10, max_value=120, value=20, step=5)

# Compute rolling (windowed) standard deviation of daily log returns,
# then annualize it by multiplying by sqrt(252) since there are ~252 trading days/year.
# `rets` was computed earlier as daily log returns for the primary ticker.
roll_vol_ann = rets[display_ticker].rolling(win).std() * np.sqrt(252)  # annualized

# Drop the initial NaNs (first `win-1` positions have insufficient data)
# and give the series a friendly name for the chart legend/hover.
roll_vol_ann = roll_vol_ann.dropna().rename("Rolling vol (ann.)")

# Build a line chart with Plotly:
# - x-axis: date index
# - y-axis: annualized volatility
# - hide x-axis title label (labels={"index": ""}) and set y-axis label text
fig = px.line(roll_vol_ann, labels={"index": "", "value": "Volatility (annualized)"})

# Clean visual style, unified hover so all points at the same date share one tooltip,
# and some padding around the plot.
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(t=40, r=20, b=40, l=60)
)

# Format y-axis as percentages (e.g., 0.25 -> "25%").
fig.update_yaxes(tickformat=".0%")

# Render the Plotly figure in Streamlit and let it expand to the container width.
st.plotly_chart(fig, use_container_width=True)


# ---- Drawdowns (peak-to-trough losses) ---------------------------------------
# Section header for the drawdown chart of the selected ticker.
st.subheader(f"Drawdowns — {display_ticker}")

# Create a simple price index that starts at 1.0 (or 100%) on the first date.
# This lets us measure losses from peaks independent of the original price level.
idx = (price_series / price_series.iloc[0]).rename("Index")

# Running maximum of the index over time (the prior peak at each date).
peak = idx.cummax()

# Drawdown = % drop from the most recent peak.
# At peaks, idx == peak → drawdown = 0. Between peaks, drawdown is negative (e.g., -0.25 = -25%).
drawdown = (idx / peak) - 1
drawdown.name = "Drawdown"

# Plot an area chart of drawdowns (fills below the line).
# Hide the x-axis title (use the section header for context) and label the y as "Drawdown".
fig = px.area(drawdown, labels={"index": "", "value": "Drawdown"})

# Clean look + unified hover tooltip + some margin padding.
fig.update_layout(template="plotly_white", hovermode="x unified",
                  margin=dict(t=40, r=20, b=40, l=60))

# Make the filled area a bit transparent so the grid and context are visible.
fig.update_traces(opacity=0.6)

# Show drawdowns as percentages (e.g., -0.18 → -18%).
fig.update_yaxes(tickformat=".0%")

# Render the chart in Streamlit and let it expand to the available width.
st.plotly_chart(fig, use_container_width=True)

# Optional KPI: the worst (most negative) drawdown over the displayed range.
max_dd = drawdown.min()
st.caption(f"Max drawdown in range: **{max_dd:.0%}**")










