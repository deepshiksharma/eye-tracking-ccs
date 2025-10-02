import numpy as np
import pandas as pd
from collections import defaultdict


def extract_emotion_rating_segments(dataframe):
    """
    Extracts each segment of emotion rating screen from the dataframe, for each stim_id.
    
    Args:
        - df (pd.DataFrame):    The dataframe containing eye-tracking data.

    Returns:
        - segs (dict): A nested dictionary containing all the segments where each emotion was rated, for each stimulus.

        segs dictionary structure:
        {
            stim_id_1: {
                emotion_1: dataframe_segment,
                emotion_2: dataframe_segment,
                ...
                emotion_n: dataframe_segment,
            },
            stim_id_2: {
                ...
            }
            ...
            stim_id_n: {
                ...
            }
        }
    """
    dataframe = dataframe.copy()

    # Keep only rows where subject was rating an emotion
    rating_rows = dataframe[dataframe['remarks'].notna() & dataframe['remarks'].str.endswith("_EMOTION_RATING")].copy()
    
    # Mark contiguous blocks
    rating_rows['_block'] = (rating_rows['remarks'] != rating_rows['remarks'].shift()).cumsum()

    segs = defaultdict(dict)

    # Group by stim_id and by contiguous block of remarks
    for (stim_id, block_id), block_df in rating_rows.groupby(['stim_id', '_block']):
        remark = block_df['remarks'].iloc[0]
        emotion = remark.replace("_EMOTION_RATING", "")
        segs[stim_id][emotion] = block_df.drop(columns=['_block'])
    
    return dict(segs)






def detect_fixations(dataframe, screen_width=1920, screen_height=1080,
                     min_duration_ms=100, dispersion_px=50, sampling_rate=1200,
                     eye="right"):
    """
    Detect fixations from Tobii gaze dataframe using a simple dispersion-threshold algorithm.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A segment of gaze data with columns:
        - 'left_gaze_point_on_display_area' or 'right_gaze_point_on_display_area'
        - 'device_time_stamp'
    screen_width : int
        Screen width in pixels.
    screen_height : int
        Screen height in pixels.
    min_duration_ms : int
        Minimum fixation duration (in ms).
    dispersion_px : int
        Maximum radius (px) for samples to be considered part of same fixation.
    sampling_rate : int
        Tobii sampling rate (Hz).
    eye : str
        Which eye to use ("left" or "right"). Default is "right".

    Returns
    -------
    dict
        {
            "fixation_count": int,
            "total_duration": float (ms),
            "mean_duration": float (ms),
            "fixations": [
                {"x": float, "y": float, "duration": float (ms),
                 "start_time": float, "end_time": float}
            ]
        }
    """
    df = dataframe.copy()

    col = f"{eye}_gaze_point_on_display_area"

    # Convert tuples -> array of shape (n, 2), filter NaNs
    coords = df[col].apply(
        lambda v: v if isinstance(v, tuple) and not any(pd.isna(v)) else (np.nan, np.nan)
    )
    coords = np.array(coords.to_list(), dtype=np.float64)

    valid_mask = ~np.isnan(coords).any(axis=1)
    coords = coords[valid_mask]
    timestamps = df.loc[valid_mask, 'device_time_stamp'].values

    if len(coords) == 0:
        return {
            "fixation_count": 0,
            "total_duration": 0.0,
            "mean_duration": 0.0,
            "fixations": []
        }

    # Convert normalized [0,1] coords → pixels
    coords[:, 0] *= screen_width
    coords[:, 1] *= screen_height

    fixations = []
    start_idx = 0
    n = len(coords)

    while start_idx < n:
        end_idx = start_idx + 1
        while end_idx < n:
            window = coords[start_idx:end_idx+1]
            x_min, y_min = window.min(axis=0)
            x_max, y_max = window.max(axis=0)
            if (x_max - x_min > dispersion_px) or (y_max - y_min > dispersion_px):
                break
            end_idx += 1

        # Compute duration (timestamps in microseconds, convert to ms)
        duration_ms = (timestamps[end_idx-1] - timestamps[start_idx]) / 1000.0
        if duration_ms >= min_duration_ms:
            fixation_coords = coords[start_idx:end_idx]
            mean_x, mean_y = fixation_coords.mean(axis=0)
            fixations.append({
                "x": mean_x,
                "y": mean_y,
                "duration": duration_ms,
                "start_time": timestamps[start_idx] / 1000.0,
                "end_time": timestamps[end_idx-1] / 1000.0
            })

        start_idx = end_idx

    fixation_count = len(fixations)
    total_duration = sum(f["duration"] for f in fixations)
    mean_duration = total_duration / fixation_count if fixation_count > 0 else 0.0

    return {
        "fixation_count": fixation_count,
        "total_duration": total_duration,
        "mean_duration": mean_duration,
        "fixations": fixations
    }


