import pandas as pd
import numpy as np
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

            
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


def get_public_holidays():
    """
    Fetch public holidays for a range of years in France.

    Parameters:
    - year_start (int): Start year for fetching holidays.
    - year_end (int): End year for fetching holidays.
    - include_alsace (bool): Include Alsace-Moselle specific holidays if True.

    Returns:
    - pd.Series: A series of public holidays as pandas datetime objects.
    """
    holidays_2020_2021 = (
        list(JoursFeries.for_year(2020).values()) +
        list(JoursFeries.for_year(2021).values())
    )
    return pd.to_datetime(holidays_2020_2021)
    

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
    X.loc[:, "year"] = X["date_x"].dt.year
    X.loc[:, "month"] = X["date_x"].dt.month
    X.loc[:, "day"] = X["date_x"].dt.day
    X.loc[:, "weekday"] = X["date_x"].dt.weekday
    X.loc[:, "hour"] = X["date_x"].dt.hour
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
    X.loc[:, "season"] = np.select(conditions, seasons, default="Unknown")
    return X

def add_indicator_features(X, school_holidays, public_holidays):
    """
    Add indicator features like holiday, weekend, lockdown, and peak hours.

    Parameters:
    X (pd.DataFrame): Input DataFrame.
    holidays (pd.Series): Series of holiday dates.
    lockdown_ranges (list): List of lockdown date ranges.

    Returns:
    pd.DataFrame: DataFrame with indicator features added.
    """
    X["school_holiday"] = X["date_x"].isin(school_holidays).astype(int)
    X["public_holiday"] = X["date_x"].isin(public_holidays).astype(int)
    X['is_peak'] = X['hour'].apply(lambda x: 1 if (6 <= x < 9 or 16 <= x < 19) else 0)
    return X

def encode_dates(X, school_holidays, public_holidays):
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
    X = add_indicator_features(X, school_holidays, public_holidays)
    X = create_cyclical_features(X, 'hour', 24)
    X = create_cyclical_features(X, 'month', 12)
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

def delete_zeros(data):
    data["truncated_date"] = data["date"].dt.floor("D")
    
    zero_count_days = (
        data.groupby(["counter_name", "truncated_date"], observed=False)["log_bike_count"]
        .sum()
        .reset_index()
        .loc[lambda x: x["log_bike_count"] == 0, ["counter_name", "truncated_date"]]
    )
    
    cleaned_data = (
        data.merge(zero_count_days, on=["counter_name", "truncated_date"], how="outer", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns=["_merge", "truncated_date"])
    )
    return cleaned_data

