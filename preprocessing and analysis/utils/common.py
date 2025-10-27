import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def overview_plot(dataframe, pupil_to_plot='left', plot_validity=True, baseline_corrected=False):
    """
    Plot the pupil diameter across the entire experiment session, with scrambled, stimulus, and fixation segments highlighted.
    
    Args:
        - dataframe (pd.DataFrame):         The dataframe containing eye-tracking data.
        - pupil_to_plot (str, optional):    Which pupil diameter to plot. Defaults to "left".
        - plot_validity (bool, optional):   Whether to plot pupil detection validity. Defaults to True.
        - baseline_corrected (bool, optional):  Whether to plot baseline corrected pupil diameter
                                                (requires input df to contain baseline corrected columns). Defaults to False.
    """
    
    plt.figure(figsize=(15, 5))
    label_pupil_dia, label_pupil_val = 'Pupil diameter (mm)', 'Pupil validity (0|1)'
    suffix = '_bc' if baseline_corrected else ''
    
    dia_col = f"{pupil_to_plot}_pupil_diameter{suffix}"
    plt.plot(dataframe[dia_col], color='k', label=label_pupil_dia)
    
    if plot_validity:
        val_col = f"{pupil_to_plot}_pupil_validity"
        plt.plot(dataframe[val_col], label=label_pupil_val)
    
    # Find contiguous segments where stim_present==True
    current_stim, current_cat, start_idx = None, None, None
    stim_segments = []

    for i, (stim_present, stim_id, stim_cat) in enumerate(zip(dataframe['stim_present'], dataframe['stim_id'], dataframe['stim_cat'])):
        if stim_present and start_idx is None:
            start_idx = i
            current_stim = stim_id
            current_cat = stim_cat
        elif not stim_present and start_idx is not None:
            stim_segments.append((start_idx, i-1, current_stim, current_cat))
            start_idx = None
            current_stim = None
            current_cat = None
    # Catch final segment
    if start_idx is not None:
        stim_segments.append((start_idx, len(dataframe)-1, current_stim, current_cat))

    # Find scrambled image segments where remarks=='SCRAMBLED IMAGE'
    start_idx = None
    scrambled_segments = []
    
    for i, remark in enumerate(dataframe['remarks']):
        if remark == 'SCRAMBLED IMAGE' and start_idx is None:
            start_idx = i
        elif remark != 'SCRAMBLED IMAGE' and start_idx is not None:
            scrambled_segments.append((start_idx, i-1))
            start_idx = None

    # Find fixation cross segments where remarks=='FIXATION CROSS'
    start_idx = None
    fix_segments = []
    for i, remark in enumerate(dataframe['remarks']):
        if remark == 'FIXATION CROSS' and start_idx is None:
            start_idx = i
        elif remark != 'FIXATION CROSS' and start_idx is not None:
            fix_segments.append((start_idx, i-1))
            start_idx = None

    # Plot stim segments, grouped by stim_cat
    colors = {}
    used_labels = set()
    for start, end, stim_id, stim_cat in stim_segments:
        if stim_cat not in colors:
            # Assign one color per category
            colors[stim_cat] = plt.cm.Pastel1(len(colors) % 8)
        label_ = stim_cat if stim_cat not in used_labels else None
        if label_ and type(label_) is str: label_ = f"{label_.capitalize()} stimuli"
        plt.axvspan(start, end, color=colors[stim_cat], alpha=0.4, label=label_)
        used_labels.add(stim_cat)

    # Plot fixation cross and scrambled image segments
    first_scrambled_label = True
    for start, end in scrambled_segments:
        label_ = 'Scrambled image' if first_scrambled_label else None
        plt.axvspan(start, end, color='black', alpha=0.1, label=label_)
        first_scrambled_label = False

    first_fix_label = True
    for start, end in fix_segments:
        label_ = 'Fixation image' if first_fix_label else None
        plt.axvspan(start, end, color='black', alpha=0.25, label=label_)
        first_fix_label = False

    plt.title(f"{pupil_to_plot.capitalize()} pupil diameter across the experiment session")
    plt.legend()
    plt.show()


def remove_invalids(dataframe, threshold=1, pad_before=150, pad_after=150, verbose=True):
    """
    Removes continuous sequences of data where the pupil is invalid for atleast `threshold` number of samples;
    as well as the data before and after that sequence (size defined by `padding`).
    
    Args:
        - dataframe (pd.DataFrame):    The dataframe containing eye-tracking data.
        - threshold (int, optional):   Minimum number of continuous invalid rows to trigger removal. Defaults to 1 (aggressive removal).
        - pad_before (int, optional):  Number of rows to remove before each invalid segment. Defaults to 150 (aggressive artifact padding).
        - pad_after (int, optional):   Number of rows to remove after each invalid segment. Defaults to 150 (aggressive artifact padding).
        - verbose (bool, optional):    Print details on how much data has been removed. Defaults to True.
    
    Returns:
        - dataframe_clean (pd.DataFrame): Dataframe after removing invalid data + padding.
    """
    dataframe = dataframe.copy()

    left_val, right_val = "left_pupil_validity", "right_pupil_validity"
    reset_index = True # reset the dataframe index after dropping rows
    n = len(dataframe)

    if left_val not in dataframe.columns or right_val not in dataframe.columns:
        raise KeyError(f"Columns not found: {left_val}, {right_val}")

    # boolean array: True where either pupil is invalid
    invalid = ((dataframe[left_val] == 0) | (dataframe[right_val] == 0)).to_numpy(dtype=int)

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
    drop_index_labels = dataframe.index[drop_positions_sorted].tolist()
    dataframe_clean = dataframe.drop(index=drop_index_labels)
    
    if reset_index:
        dataframe_clean = dataframe_clean.reset_index(drop=True)

    if verbose:
        print("input dataframe size:", dataframe.shape[0])
        print("ouput dataframe size:", dataframe_clean.shape[0])
        print("number of segments removed:", num_segments)
        print("total number of rows removed:", len(drop_positions_sorted))
    
    return dataframe_clean


def replace_invalids_w_nans(dataframe, threshold=1, pad_before=150, pad_after=150, verbose=True):
    """
    Replaces continuous sequences of data where the pupil is invalid for at least `threshold` samples
    with NaN values, including the data before and after that sequence (size defined by padding).

    Args:
        - dataframe (pd.DataFrame):    The dataframe containing eye-tracking data.
        - threshold (int, optional):   Minimum number of continuous invalid rows to trigger NaN replacement. Defaults to 1 (aggressive replacement).
        - pad_before (int, optional):  Number of rows to replace with NaN before each invalid segment. Defaults to 150 (aggressive artifact padding).
        - pad_after (int, optional):   Number of rows to replace with NaN after each invalid segment. Defaults to 150 (aggressive artifact padding).
        - verbose (bool, optional):    Print details on how much data has been repalced. Defaults to True.
    
    Returns:
        - dataframe (pd.DataFrame): Dataframe with invalid data and padding replaced by NaN.
    """
    dataframe = dataframe.copy()
    
    left_val, right_val = "left_pupil_validity", "right_pupil_validity"
    n = len(dataframe)

    if left_val not in dataframe.columns or right_val not in dataframe.columns:
        raise KeyError(f"Columns not found: {left_val}, {right_val}")

    # boolean array: True where either pupil is invalid
    invalid = ((dataframe[left_val] == 0) | (dataframe[right_val] == 0)).to_numpy(dtype=int)

    # find contiguous invalid stretches
    padded = np.concatenate(([0], invalid, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    num_segments = 0
    replace_positions = set()
    for st, ed in zip(starts, ends):
        seg_len = ed - st + 1
        if seg_len >= threshold:
            num_segments += 1  # count this segment
            # mark segment itself
            replace_positions.update(range(st, ed + 1))
            # mark padding before/after
            pre_start = max(0, st - pad_before)
            post_end = min(n - 1, ed + pad_after)
            replace_positions.update(range(pre_start, st))
            replace_positions.update(range(ed + 1, post_end + 1))

    replace_positions_sorted = sorted(replace_positions)

    # Replace all target rows with NaN across all columns
    dataframe.iloc[replace_positions_sorted] = np.nan

    if verbose:
        print("dataframe size:", dataframe.shape[0])
        print("number of segments replaced with np.nan:", num_segments)
        print("total number of rows replaced with np.nan:", len(replace_positions_sorted))
    
    return dataframe


def extract_emotion_rating_segments(dataframe):
    """
    Extracts each segment of emotion rating screen from the dataframe, for each stim_id.
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.

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


def extract_stim_viewing_segments(dataframe):
    """
    Extracts all rows where stim_present == True, for each stim_id.

    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.

    Returns:
        - segs (dict): A dictionary containing the segment where each stimulus was viewed.

        segs dictionary structure:
        {
            stim_id_1: dataframe_segment,
            stim_id_2: dataframe_segment,
            ...
        }
    """
    dataframe = dataframe.copy()

    # Keep only rows where stim_present==True
    stim_rows = dataframe[dataframe['stim_present'] == True].copy()

    segs = {}
    for stim_id, stim_df in stim_rows.groupby('stim_id'):
        segs[stim_id] = stim_df
    
    return segs


def convert_to_pygaze_compatible_format(dataframe):
    """
    Convert data captured by Tobii into a format compatible with PyGaze to calculate fixations and saccades.
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.
    Returns:
        - x, y, z
    """
    dataframe = dataframe.copy()

    # display resolution
    screen_w, screen_h = 1920, 1080

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
