import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def overview_plot(dataframe, pupil_to_plot=None):
    """
    Plot the pupil diameter across the entire experiment session, with scrambled, stimulus, and fixation segments highlighted.
    
    Args:
        - dataframe (pd.DataFrame):    The dataframe containing eye-tracking data.
        - pupil_to_plot (str):  Which pupil diameter to plot. Allowed values: "left" or "right".

    Returns:
        - 1 (int): Will happen if you forget to pass in the `pupil_to_plot` argument.
    """

    if pupil_to_plot not in ['left', 'right']:
        print("Specify which pupil to plot (\"left\" or \"right\") with the `pupil_to_plot` argument.")
        return 1
    
    plt.figure(figsize=(15, 5))
    if pupil_to_plot=='left':
        plt.plot(dataframe['left_pupil_diameter'], color='k')
        plt.plot(dataframe['left_pupil_validity'], label='Pupil validity')
    elif pupil_to_plot=='right':
        plt.plot(dataframe['right_pupil_diameter'], color='k')
        plt.plot(dataframe['right_pupil_validity'], label='Pupil validity')

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
        plt.axvspan(start, end, color=colors[stim_cat], alpha=0.4, label=label_)
        used_labels.add(stim_cat)

    # Plot fixation cross and scrambled image segments
    first_scrambled_label = True
    for start, end in scrambled_segments:
        label_ = 'scrambled' if first_scrambled_label else None
        plt.axvspan(start, end, color='black', alpha=0.1, label=label_)
        first_scrambled_label = False

    first_fix_label = True
    for start, end in fix_segments:
        label_ = 'Fix' if first_fix_label else None
        plt.axvspan(start, end, color='black', alpha=0.25, label=label_)
        first_fix_label = False

    plt.ylabel('Pupil diameter (mm)')
    plt.title(f"{pupil_to_plot.capitalize()} pupil diameter across the experiment session")
    plt.legend()
    plt.show()


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
