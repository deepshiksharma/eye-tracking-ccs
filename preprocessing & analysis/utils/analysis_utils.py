import matplotlib.pyplot as plt
from collections import defaultdict
from preprocess_utils import pygaze_compatible_format_fixsacc
from pygazeanalyser.detectors import fixation_detection


def overview_plot(dataframe, pupil_to_plot=None):
    """
    Plot the pupil diameter across the entire experiment session, with scrambled, stimulus, and fixation segments highlighted.
    
    Args:
        - df (pd.DataFrame):    The dataframe containing eye-tracking data.
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
        - df (pd.DataFrame): The dataframe containing eye-tracking data.

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


def is_inside_box(x, y, x1, y1, x2, y2):
    """
    Args: x, y, x1, y1, x2, y2
    """
    return x1 <= x < x2 and y1 <= y <= y2

def check_gaze_fixation_on_button(x, y):
    """
    Args: x, y
    """
    if is_inside_box(x, y, 120, 400, 360, 850):
        return 1
    elif is_inside_box(x, y, 360, 400, 600, 850):
        return 2
    elif is_inside_box(x, y, 600, 400, 840, 850):
        return 3
    elif is_inside_box(x, y, 840, 400, 1080, 850):
        return 4
    elif is_inside_box(x, y, 1080, 400, 1320, 850):
        return 5
    elif is_inside_box(x, y, 1320, 400, 1560, 850):
        return 6
    elif is_inside_box(x, y, 1560, 400, 1800, 850):
        return 7
    else:
        return None

def get_max_fixation_on_button(dataframe):
    """
    ...
    """
    # convert df to pygaze compatible format
    x, y, t = pygaze_compatible_format_fixsacc(dataframe)
    
    # get fixations
    Sfix, Efix = fixation_detection(x, y, t)
          # A single Efix list is of the format:
          # [starttime, endtime, duration, endx, endy]

    # get the longest fixation duration
    max_fix_val, max_fix_idx = 0, 0
    for i in Efix:
        if i[2] >= max_fix_val:
            max_fix_val = i[2]
            max_fix_idx = Efix.index(i)
    
    # get which rating button corresponds to x, y co-ordinates of longest fixation
    fix_button = 0  # default
    try:
        if Efix and 0 <= max_fix_idx < len(Efix):
            detected_fix = Efix[max_fix_idx]
            if len(detected_fix) == 5:  # ensure it has all elements
                fix_button = check_gaze_fixation_on_button(detected_fix[-2], detected_fix[-1])
    except Exception as e:
        print(f"Warning: could not get fixation button: {e}")

    return fix_button


def get_max_2_fixation_on_button(dataframe):
    """
    Returns the buttons corresponding to the longest and 2nd longest fixations.
    Adds error handling to avoid index errors and empty fixation lists.
    """
    fix1_button, fix2_button = 0, 0  # defaults
    
    try:
        # convert df to pygaze compatible format
        x, y, t = pygaze_compatible_format_fixsacc(dataframe)
        
        # get fixations
        Sfix, Efix = fixation_detection(x, y, t)
        # Each Efix list: [starttime, endtime, duration, endx, endy]

        if not Efix:
            return fix1_button, fix2_button  # no fixations available

        # Sort fixations by duration descending
        sorted_fix = sorted(Efix, key=lambda i: i[2], reverse=True)

        # Longest fixation
        if len(sorted_fix) >= 1:
            detected_fix = sorted_fix[0]
            if len(detected_fix) >= 5:
                fix1_button = check_gaze_fixation_on_button(detected_fix[-2], detected_fix[-1])

        # 2nd longest fixation
        if len(sorted_fix) >= 2:
            detected_fix2 = sorted_fix[1]
            if len(detected_fix2) >= 5:
                fix2_button = check_gaze_fixation_on_button(detected_fix2[-2], detected_fix2[-1])

    except Exception as e:
        print(f"Warning: could not compute fixation buttons: {e}")

    return fix1_button, fix2_button


def get_max_n_fixation_on_button(dataframe):
    """
    Returns the buttons corresponding to the longest and 2nd longest fixations.
    Adds error handling to avoid index errors and empty fixation lists.
    """
    fix1_button, fix2_button, fix3_button, fix4_button, fix5_button = 0, 0, 0, 0, 0  # defaults
    
    try:
        # convert df to pygaze compatible format
        x, y, t = pygaze_compatible_format_fixsacc(dataframe)
        
        # get fixations
        Sfix, Efix = fixation_detection(x, y, t)
        # Each Efix list: [starttime, endtime, duration, endx, endy]

        if not Efix:
            return fix1_button, fix2_button  # no fixations available

        # Sort fixations by duration descending
        sorted_fix = sorted(Efix, key=lambda i: i[2], reverse=True)

        # Longest fixation
        if len(sorted_fix) >= 1:
            detected_fix = sorted_fix[0]
            if len(detected_fix) >= 5:
                fix1_button = check_gaze_fixation_on_button(detected_fix[-2], detected_fix[-1])

        # 2nd longest fixation
        if len(sorted_fix) >= 2:
            detected_fix2 = sorted_fix[1]
            if len(detected_fix2) >= 5:
                fix2_button = check_gaze_fixation_on_button(detected_fix2[-2], detected_fix2[-1])

        # 3rd longest fixation
        if len(sorted_fix) >= 3:
            detected_fix3 = sorted_fix[2]
            if len(detected_fix3) >= 5:
                fix3_button = check_gaze_fixation_on_button(detected_fix3[-2], detected_fix3[-1])

        # 4th longest fixation
        if len(sorted_fix) >= 4:
            detected_fix4 = sorted_fix[3]
            if len(detected_fix4) >= 5:
                fix4_button = check_gaze_fixation_on_button(detected_fix4[-2], detected_fix4[-1])

        # 5th longest fixation
        if len(sorted_fix) >= 5:
            detected_fix5 = sorted_fix[4]
            if len(detected_fix5) >= 5:
                fix5_button = check_gaze_fixation_on_button(detected_fix5[-2], detected_fix5[-1])

    except Exception as e:
        print(f"Warning: could not compute fixation buttons: {e}")

    return fix1_button, fix2_button, fix3_button, fix4_button, fix5_button
