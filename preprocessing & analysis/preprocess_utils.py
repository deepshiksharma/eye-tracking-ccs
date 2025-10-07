import ast
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def remove_invalids(df, threshold=50, pad_before=150, pad_after=150):
    """
    Removes continuous sequences of data where the pupil is invalid for atleast `threshold` number of samples;
    as well as the data before and after that sequence (size defined by `padding`).
    
    Args:
        - df (pd.DataFrame): The dataframe containing eye-tracking data.
        - threshold (int, optional):  Minimum number of continuous invalid rows to trigger removal. Defaults to 50.
        - pad_before (int, optional): Number of rows to remove before each invalid segment. Defaults to 150.
        - pad_after (int, optional):  Number of rows to remove after each invalid segment. Defaults to 150.

    Returns:
        - df_clean (pd.DataFrame): Dataframe after removing invalid data + padding.
    """
    
    left_val, right_val = "left_pupil_validity", "right_pupil_validity"
    reset_index = True # reset the dataframe index after dropping rows
    n = len(df)

    if left_val not in df.columns or right_val not in df.columns:
        raise KeyError(f"Columns not found: {left_val}, {right_val}")

    # boolean array: True where either pupil is invalid
    invalid = ((df[left_val] == 0) | (df[right_val] == 0)).to_numpy(dtype=int)

    # find contiguous invalid stretches
    padded = np.concatenate(([0], invalid, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    num_segments = 0
    drop_positions = set()
    for st, ed in zip(starts, ends):
        seg_len = ed - st + 1
        if seg_len >= threshold:
            num_segments += 1  # count this segment
            # drop segment itself
            drop_positions.update(range(st, ed + 1))
            # drop padding before/after
            pre_start = max(0, st - pad_before)
            post_end = min(n - 1, ed + pad_after)
            drop_positions.update(range(pre_start, st))
            drop_positions.update(range(ed + 1, post_end + 1))

    drop_positions_sorted = sorted(drop_positions)
    drop_index_labels = df.index[drop_positions_sorted].tolist()
    df_clean = df.drop(index=drop_index_labels)
    
    if reset_index:
        df_clean = df_clean.reset_index(drop=True)

    print("input dataframe size:", df.shape[0])
    print("ouput dataframe size:", df_clean.shape[0])
    print("number of segments removed:", num_segments)
    print("total number of rows removed:", len(drop_positions_sorted))
    
    return df_clean


def apply_smoothing(df, window_length=101, polyorder=3, plot_fig=False):
    """
    Applies smoothing to the pupil diameter to decrease random fluctuations.
    
    Args:
        - df (pd.DataFrame): The dataframe containing eye-tracking data,
                             ideally after invalid rows have been removed by the `remove_invalids` function.
        - window_length (int, optional): The size of moving window over which the polynomial is fitted. Argument is passed to Savitzky-Golay filter. Defaults to 101.
        - polyorder (int, optional):     The degree of polynomial used to approximate the data inside each window. Argument is passed to Savitzky-Golay filter. Defaults to 3.
        - plot_fig (bool, optional):     If True, plot figure which overlays the raw and smoothed signals. Defaults to False.
    
    Returns:
        - df (pd.DataFrame): Dataframe after smoothing both columns containing the pupil diameter.
                             This dataframe will contain two new columns: "left_pupil_diameter_smooth" and "right_pupil_diameter_smooth".
    """
    
    # This function applies the Savitzky-Golay filter to smooth a column containing the pupil diameter.
    def apply_smoothing_per_pupil(pupil_diameter_column):
        x = np.arange(len(pupil_diameter_column))
        y = signal.savgol_filter(pupil_diameter_column, window_length=window_length, polyorder=polyorder)
        return x, y

    left = df.left_pupil_diameter
    right = df.right_pupil_diameter

    leftx, lefty = apply_smoothing_per_pupil(left)
    rightx, righty = apply_smoothing_per_pupil(right)

    if plot_fig == True:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(left, color='b', alpha=0.5, label='Raw signal')
        plt.plot(leftx, lefty, color='r', label='Smoothed signal')
        plt.title("Left pupil diameter")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(right, color='b', alpha=0.5, label='Raw signal')
        plt.plot(rightx, righty, color='r', label='Smoothed signal')
        plt.title("Right pupil diameter")
        plt.legend()
        plt.tight_layout()
        plt.show()

    df["left_pupil_diameter_smooth"] = lefty
    df["right_pupil_diameter_smooth"] = righty
    
    return df


def baseline_correction(df):
    """
    Performs subtractive baseline correction (per pupil, per trial) by taking the mean from a 500 ms baseline period (scrambled image).
    
    Args:
        - df (pd.DataFrame): The dataframe containing eye-tracking data,
                             ideally after removing invalid rows using `remove_invalids`,
                             and smoothing the pupil diameter signal using `apply_smoothing`.

    Returns:
        - df (pd.DataFrame): Dataframe after performing baseline correction.
                             This dataframe will contain two new columns: "left_pupil_dia_bc" and "right_pupil_dia_bc".
    """

    # columns for baseline-corrected values
    df["left_pupil_dia_bc"] = np.nan
    df["right_pupil_dia_bc"] = np.nan

    # group by stim_id (one trial per image)
    for stim_id, trial_df in df.groupby("stim_id"):
        if pd.isna(stim_id):
            continue
        
        # scrambled image rows before the stimulus
        scram_rows = trial_df[trial_df["remarks"] == "SCRAMBLED IMAGE"]
        if scram_rows.empty:
            continue
        
        # 500 ms baseline
        baseline_rows = scram_rows.tail(1000).head(600)

        left_base = baseline_rows["left_pupil_dia_smooth"].mean()
        right_base = baseline_rows["right_pupil_dia_smooth"].mean()
        
        # all rows after scrambled image until the next fixation cross
        mask = (df["stim_id"] == stim_id) & (df["remarks"] != "SCRAMBLED IMAGE")
        
        df.loc[mask, "left_pupil_dia_bc"] = (
            df.loc[mask, "left_pupil_dia_smooth"] - left_base
        )
        df.loc[mask, "right_pupil_dia_bc"] = (
            df.loc[mask, "right_pupil_dia_smooth"] - right_base
        )

    return df


def drop_scrambled(df):
    """
    Drop all rows where df["remarks"]=="SCRAMBLED IMAGE"
    Args:
        - df (pd.DataFrame): The dataframe containing eye-tracking data.
    Returns:
        - df (pd.DataFrame): Dataframe after removing all rows corresponding to scrambled images.
    """
    df = df[df["remarks"] != "SCRAMBLED IMAGE"].copy()
    return df


def pygaze_compatible_format_fixsacc(df):
    """
    Convert data captured by Tobii into a format compatible with PyGaze to calculate fixations and saccades.
    Args:
        - df (pd.DataFrame): The dataframe containing eye-tracking data.
    Returns:
        - data (np.array): PyGaze compatible np array ([[t, x, y], [t, x, y], ...]).
    """
    # display resolution
    screen_w, screen_h = 1920, 1080

    def parse_xy(s):
        try:
            return ast.literal_eval(s)
        except:
            return (np.nan, np.nan)

    # Parse tuples
    df["lx"], df["ly"] = zip(*df["left_gaze_point_on_display_area"].map(parse_xy))
    df["rx"], df["ry"] = zip(*df["right_gaze_point_on_display_area"].map(parse_xy))

    # Convert normalized coordinates to pixels
    lx = np.where(df["left_gaze_point_validity"]==1, df["lx"]*screen_w, np.nan)
    ly = np.where(df["left_gaze_point_validity"]==1, df["ly"]*screen_h, np.nan)
    rx = np.where(df["right_gaze_point_validity"]==1, df["rx"]*screen_w, np.nan)
    ry = np.where(df["right_gaze_point_validity"]==1, df["ry"]*screen_h, np.nan)

    # Average both eyes
    x = np.nanmean(np.column_stack([lx, rx]), axis=1)
    y = np.nanmean(np.column_stack([ly, ry]), axis=1)
    
    # Convert timestamps to ms, relative to start
    t = (df["system_time_stamp"] - df["system_time_stamp"].iloc[0]) / 1000.0

    # Drop NaN samples
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    t = t[mask]
    
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    return x, y, t
