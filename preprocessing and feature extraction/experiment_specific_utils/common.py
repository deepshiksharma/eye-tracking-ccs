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


def extract_emotion_rating_segments(dataframe):
    """
    Extracts each segment of emotion rating screen from the dataframe, for each stim_id.
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.

    Returns:
        - dict: A nested dictionary containing all the segments where each emotion was rated, for each stimulus.

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
        - dict: A dictionary containing the segment where each stimulus was viewed.

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
