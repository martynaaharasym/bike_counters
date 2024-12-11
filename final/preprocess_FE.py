import pandas as pd
import numpy as np
from vacances_scolaires_france import SchoolHolidayDates
            
def get_zone_c_holidays():
    """
    Fetch holidays for Zone C for the years 2020 and 2021.
    Returns a combined list of holidays as pandas datetime objects.
    """
    holiday_dates = SchoolHolidayDates()
    zone_c_holidays_2020 = holiday_dates.holidays_for_year_and_zone(2020, 'C')
    zone_c_holidays_2021 = holiday_dates.holidays_for_year_and_zone(2021, 'C')
    
    all_zone_c_holidays = list(zone_c_holidays_2020.keys()) + list(zone_c_holidays_2021.keys())
    return pd.to_datetime(all_zone_c_holidays)

def curfew_periods(df, date_column="date_x"):
    """
    Add a binary column indicating whether a timestamp falls within curfew hours.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing datetime data.
    date_column (str): Name of the column containing datetime data.

    Returns:
    pd.DataFrame: DataFrame with a 'curfew' column added.
    """
    curfews = [
        ("2020-10-17", "2020-10-29", 21, 6),
        ("2021-01-16", "2021-03-20", 18, 6),
        ("2021-03-21", "2021-05-19", 19, 6),
        ("2021-05-20", "2021-06-09", 21, 6),
        ("2021-06-10", "2021-06-20", 23, 6)
    ]

    def is_curfew(timestamp):
        for start_date, end_date, start_hour, end_hour in curfews:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            if start <= timestamp <= end:
                hour = timestamp.hour
                if hour >= start_hour or hour < end_hour:
                    return 1
        return 0

    df["curfew"] = df[date_column].apply(is_curfew)
    return df


def create_cyclical_features(data, column, period):
    """
    Create cyclical features (sine and cosine) for a given column and drop the original column.

    Parameters:
    data (pd.DataFrame): Input DataFrame.
    column (str): Column name to create cyclical features for.
    period (int): Period of the cycle (e.g., 24 for hours, 12 for months).

    Returns:
    pd.DataFrame: DataFrame with added cyclical features and the original column dropped.
    """
    data[f'sin_{column}'] = np.sin(2 * np.pi * data[column] / period)
    data[f'cos_{column}'] = np.cos(2 * np.pi * data[column] / period)
    return data.drop(columns=[column])

def add_basic_date_features(X):
    """
    Add basic date features like year, month, day, weekday, and hour.

    Parameters:
    X (pd.DataFrame): Input DataFrame containing a 'date_x' column.

    Returns:
    pd.DataFrame: DataFrame with basic date features added.
    """
    X["year"] = X["date_x"].dt.year
    X["month"] = X["date_x"].dt.month
    X["day"] = X["date_x"].dt.day
    X["weekday"] = X["date_x"].dt.weekday
    X["hour"] = X["date_x"].dt.hour
    return X

def add_season_feature(X):
    """
    Add a season feature based on the month.

    Parameters:
    X (pd.DataFrame): Input DataFrame containing a 'month' column.

    Returns:
    pd.DataFrame: DataFrame with a 'season' column added.
    """
    conditions = [
        (X["month"].isin([12, 1, 2])),  # Winter
        (X["month"].isin([3, 4, 5])),   # Spring
        (X["month"].isin([6, 7, 8])),   # Summer
        (X["month"].isin([9, 10, 11]))  # Fall
    ]
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    X["season"] = np.select(conditions, seasons, default="Unknown")
    return X

def add_indicator_features(X, holidays):
    """
    Add indicator features like holiday, weekend, lockdown, and peak hours.

    Parameters:
    X (pd.DataFrame): Input DataFrame.
    holidays (pd.Series): Series of holiday dates.
    lockdown_ranges (list): List of lockdown date ranges.

    Returns:
    pd.DataFrame: DataFrame with indicator features added.
    """
    X["holiday"] = X["date_x"].isin(holidays).astype(int)
    X["weekend"] = (X["date_x"].dt.dayofweek > 4).astype(int)
    X['is_peak'] = X['hour'].apply(lambda x: 1 if (6 <= x < 9 or 16 <= x < 19) else 0)
    return X

def encode_dates(X, holidays):
    """
    Encode date information by adding various features.

    Parameters:
    X (pd.DataFrame): Input DataFrame containing a 'date_x' column.
    holidays (pd.Series): Series of holiday dates.

    Returns:
    pd.DataFrame: DataFrame with all date-related features added.
    """

    X = add_basic_date_features(X)
    X = add_season_feature(X)
    X = add_indicator_features(X, holidays)
    X = create_cyclical_features(X, 'hour', 24)
    X = create_cyclical_features(X, 'month', 12)
    X = create_cyclical_features(X, 'weekday', 7)
    X = curfew_periods(X)


    return X.drop(columns=['date_x'])

def categorize_weather(data):
    """
    Categorize weather data into rain and snow categories.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing weather-related columns.

    Returns:
    pd.DataFrame: DataFrame with weather categories added.
    """
    data['rain_category'] = pd.cut(
        data['rr1'], bins=[-1, 0, 2, 10, float('inf')],
        labels=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    )
    # data['snow_category'] = pd.cut(
    #    data['ht_neige'], bins=[-1, 0, 0.01, 0.05, float('inf')],
    #    labels=['No Snow', 'Light Snow', 'Moderate Snow', 'Heavy Snow']
    # )
    return data

def add_weather_indicators(data):
    """
    Add binary indicators for extreme weather conditions.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing weather-related columns.

    Returns:
    pd.DataFrame: DataFrame with weather indicators added.
    """
    data['is_hot_day'] = (data['t'] > 300).astype(int)  # Assuming temperature in Kelvin
    data['is_cold_day'] = (data['t'] < 283).astype(int)
    data['high_wind'] = (data['ff'] > 5).astype(int)
    return data

def engineer_weather_features(data):
    """
    Engineer weather-related features from weather data.

    Parameters:
    data (pd.DataFrame): Input DataFrame containing weather-related columns.

    Returns:
    pd.DataFrame: DataFrame with all weather-related features added.
    """
    data = categorize_weather(data)
    data = add_weather_indicators(data)
    return data

def remove_outliers(data):
    data["date_truncated"] = data["date"].dt.floor("D")

    cleaned_data = (
        data.groupby(["counter_name", "date_truncated"])
        ["log_bike_count"].sum()
        .to_frame()
        .reset_index()
        .query("log_bike_count == 0")
        [["counter_name", "date_truncated"]]
        .merge(data, on=["counter_name", "date_truncated"], how="right", indicator=True)
        .query("_merge == 'right_only'")
        .drop(columns=["_merge", "date_truncated"])
    )

    return cleaned_data