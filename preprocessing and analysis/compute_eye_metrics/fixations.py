import ast
import numpy as np
import pandas as pd
from .pygazeanalyser_fixations import fixation_detection


def convert_to_pygaze_compatible_format(dataframe, screen_w=1920, screen_h=1080):
    """
    Convert data captured by Tobii into a format compatible with PyGaze.
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.
        - screen_w (int, optional): Width of the screen in pixels. Defaults to 1920.
        - screen_h (int, optional): Height of the screen in pixels. Defaults to 1080.
    Returns:
        - x, y, z (np.array):   Gaze co-ordinates x, y, and the timestamps t.
    """
    dataframe = dataframe.copy()

    def parse_xy(s):
        try:
            return ast.literal_eval(s)
        except:
            return (np.nan, np.nan)

    # Parse tuples
    dataframe["lx"], dataframe["ly"] = zip(*dataframe["left_gaze_point_on_display_area"].map(parse_xy))
    dataframe["rx"], dataframe["ry"] = zip(*dataframe["right_gaze_point_on_display_area"].map(parse_xy))

    # Convert normalized coordinates to pixels
    lx = np.where(dataframe["left_gaze_point_validity"]==1, dataframe["lx"]*screen_w, np.nan)
    ly = np.where(dataframe["left_gaze_point_validity"]==1, dataframe["ly"]*screen_h, np.nan)
    rx = np.where(dataframe["right_gaze_point_validity"]==1, dataframe["rx"]*screen_w, np.nan)
    ry = np.where(dataframe["right_gaze_point_validity"]==1, dataframe["ry"]*screen_h, np.nan)

    # Average both eyes
    x = np.nanmean(np.column_stack([lx, rx]), axis=1)
    y = np.nanmean(np.column_stack([ly, ry]), axis=1)
    
    # Convert timestamps to ms, relative to start
    t = (dataframe["system_time_stamp"] - dataframe["system_time_stamp"].iloc[0]) / 1000.0
    
    # Drop NaN samples
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    t = t[mask]
    
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    return x, y, t


def detect_fixations(dataframe, missing=0.0, maxdist=25, mindur=100):
    """
    Extracts fixation metrics from a given dataframe.
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.
        - missing, maxdist, mindur: Various optional parameters for fixation detection.
    
    Returns:
        - pd.DataFrame: Dataframe containing all detected fixations, along with their corresponding metrics.
    """
    x, y, t = convert_to_pygaze_compatible_format(dataframe)
    _, Efix = fixation_detection(x, y, t, missing=missing, maxdist=maxdist, mindur=mindur)
    
    if not Efix:
        return pd.DataFrame(columns=["start_time_ms", "end_time_ms", "duration_ms", "end_x_px", "end_y_px"])
    
    df = pd.DataFrame(Efix, columns=["start_time_ms", "end_time_ms", "duration_ms", "end_x_px", "end_y_px"])
    return df
