import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import holidays # A useful library for standard holidays
import os

# --- 1. Load and Prepare Data ---
# The CSV ships next to this script.
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "messy_mixed_timeseries_data.csv")
df = pd.read_csv(file_path)

print("--- Initial DataFrame Head ---")
print(df.head())
print("\n--- Initial DataFrame Info ---")
df.info()

# Required columns for Prophet: 'ds' (datestamp) and 'y' (numerical value)

# Convert 'ds' column to datetime, coercing errors
print("\n--- Cleaning 'ds' column ---")
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Drop rows where 'ds' conversion failed (NaT values)
initial_rows = len(df)
df.dropna(subset=['ds'], inplace=True)
rows_after_ds_clean = len(df)
print(f"Removed {initial_rows - rows_after_ds_clean} rows with invalid dates in 'ds' column.")

# Handle missing values in 'y' column by interpolation
print("--- Cleaning 'y' column (Interpolating missing values) ---")
df['y'] = df['y'].interpolate()
# After interpolation, if first/last values were NaN, they might remain. Drop any remaining NaNs.
df.dropna(subset=['y'], inplace=True)
rows_after_y_clean = len(df)
print(f"Removed {rows_after_ds_clean - rows_after_y_clean} rows after 'y' interpolation and dropna.")


# Ensure 'ds' column is unique (Prophet prefers unique datestamps)
# If duplicates exist, aggregate 'y' values (e.g., mean) for each unique 'ds'
if not df['ds'].is_unique:
    print("\n--- Aggregating duplicate 'ds' entries ---")
    df = df.groupby('ds')['y'].mean().reset_index()

# Sort by 'ds' to ensure chronological order
df = df.sort_values(by='ds').reset_index(drop=True)


print("\n--- Cleaned DataFrame Head (ready for Prophet) ---")
print(df.head())
print("\n--- Cleaned DataFrame Info ---")
df.info()


# --- 2. Define and Tune Holidays ---
# Prophet can incorporate custom holidays. You asked for 2-3.
# Let's define some common holidays. You can customize these based on your data's context.

# Example: Define a couple of specific holidays
# Format: DataFrame with 'holiday', 'ds' (date), 'lower_window', 'upper_window'
# lower_window and upper_window extend the holiday effect before/after the specified date.

# We'll use the 'holidays' library for convenience, assuming your data spans a few years.
# Get years present in your data to define holidays relevant to your data range.
min_year = df['ds'].min().year
max_year = df['ds'].max().year
# Include a year before and after data range for robust holiday definition
years_to_consider = list(range(min_year - 1, max_year + 2))

# Using 'holidays' library to get standard holidays (e.g., US holidays)
# You can change 'US' to 'IN' for India, or another country code as needed (e.g., holidays.India(years=...))
us_holidays = holidays.US(years=years_to_consider)

# Convert holidays to a DataFrame in Prophet's required format
my_holidays = pd.DataFrame([
    {'holiday': name, 'ds': date} for date, name in sorted(us_holidays.items())
])

# Filter to get only 2-3 specific holidays for tuning (as per your request)
# You can choose based on names or importance. These are common examples.
selected_holidays_names = ['New Year\'s Day', 'Christmas Day', 'Thanksgiving']
my_holidays = my_holidays[my_holidays['holiday'].isin(selected_holidays_names)]

# Add lower_window and upper_window if you want the holiday effect to span multiple days
# For example, Christmas effect might start a day before, or Thanksgiving might be a long weekend.
my_holidays['lower_window'] = 0
my_holidays['upper_window'] = 0
my_holidays.loc[my_holidays['holiday'] == 'Christmas Day', 'lower_window'] = -1 # Example: Christmas effect starts a day before
my_holidays.loc[my_holidays['holiday'] == 'Thanksgiving', 'lower_window'] = -1 # Example: Thanksgiving effect starts a day before
my_holidays.loc[my_holidays['holiday'] == 'Thanksgiving', 'upper_window'] = 1 # Example: Thanksgiving effect extends a day after


# Convert 'ds' in holidays DataFrame to datetime
my_holidays['ds'] = pd.to_datetime(my_holidays['ds'])


print("\n--- Defined and Filtered Holidays for Tuning ---")
print(my_holidays)


# --- 3. Apply Prophet Algorithm ---
# Initialize Prophet model with daily, weekly, yearly seasonality, and custom holidays
# You can experiment with seasonality_mode ('additive' or 'multiplicative')
# and explicitly set seasonality if your data has clear daily, weekly, or yearly patterns.
model = Prophet(
    holidays=my_holidays, # Include defined holidays
    daily_seasonality=True, # Set to True if your data has daily patterns
    weekly_seasonality=True, # Set to True if your data has weekly patterns
    yearly_seasonality=True, # Set to True if your data has yearly patterns
    seasonality_mode='additive' # 'additive' is common for constant seasonality effect
                               # 'multiplicative' for seasonality that grows with trend
)

# Fit the model to your cleaned data
print("\n--- Fitting Prophet Model ---")
model.fit(df)
print("Prophet Model Fit Complete.")

# --- 4. Make Future Predictions ---
# Create a DataFrame with future dates for forecasting
# For example, forecast for the next 30 days
future = model.make_future_dataframe(periods=30)
print("\n--- Future DataFrame Head ---")
print(future.head())

# Make predictions
forecast = model.predict(future)

print("\n--- Forecast Head ---")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())


# --- 5. Plotting the Forecast ---

print("\n--- Generating Plots ---")

# Plotting with Matplotlib (simple plot)
fig, ax = plt.subplots(figsize=(15, 7))
model.plot(forecast, ax=ax)
ax.set_title('Prophet Forecast with Holidays (Matplotlib)')
ax.set_xlabel('Date')
ax.set_ylabel('Value (y)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting components (trend, seasonality, holidays) with Matplotlib
# This shows the individual effects of trend, weekly, yearly, and holidays
fig_components = model.plot_components(forecast)
fig_components.suptitle('Prophet Forecast Components (Matplotlib)', y=1.02) # Adjust title position
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
plt.show()


# Plotting with Plotly (interactive plot)
# This requires 'plotly' library for interactive visualization
try:
    fig_plotly = plot_plotly(model, forecast)
    fig_plotly.update_layout(title_text='Prophet Forecast with Holidays (Interactive Plotly)')
    fig_plotly.show()
except ImportError:
    print("Plotly not installed. Skipping interactive forecast plot. Install with 'pip install plotly'")

# Plotting components (trend, seasonality, holidays) with Plotly
try:
    fig_components_plotly = plot_components_plotly(model, forecast)
    fig_components_plotly.update_layout(title_text='Prophet Forecast Components (Interactive Plotly)')
    fig_components_plotly.show()
except ImportError:
    print("Plotly not installed. Skipping interactive components plot. Install with 'pip install plotly'")


print("\nProphet algorithm code generated. Please run this in your local environment.")
#Authorshwet