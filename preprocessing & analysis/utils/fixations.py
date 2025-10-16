import os
import numpy as np
import pandas as pd
from pygazeanalyser.detectors import fixation_detection
from .common import convert_to_pygaze_compatible_format, extract_stim_viewing_segments, extract_emotion_rating_segments
from collections import defaultdict


def is_inside_box(x, y, x1, y1, x2, y2):
    """
    Args: x, y, x1, y1, x2, y2
    """
    return x1 <= x < x2 and y1 <= y <= y2

def check_fixation_on_button(x, y):
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


def get_top_5_fixation_on_button(dataframe):
    """
    Returns the buttons corresponding to the longest and 2nd longest fixations.
    Adds error handling to avoid index errors and empty fixation lists.
    """
    fix1_button, fix2_button, fix3_button, fix4_button, fix5_button = 0, 0, 0, 0, 0  # defaults
    
    try:
        # convert df to pygaze compatible format
        x, y, t = convert_to_pygaze_compatible_format(dataframe)
        
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
                fix1_button = check_fixation_on_button(detected_fix[-2], detected_fix[-1])

        # 2nd longest fixation
        if len(sorted_fix) >= 2:
            detected_fix2 = sorted_fix[1]
            if len(detected_fix2) >= 5:
                fix2_button = check_fixation_on_button(detected_fix2[-2], detected_fix2[-1])

        # 3rd longest fixation
        if len(sorted_fix) >= 3:
            detected_fix3 = sorted_fix[2]
            if len(detected_fix3) >= 5:
                fix3_button = check_fixation_on_button(detected_fix3[-2], detected_fix3[-1])

        # 4th longest fixation
        if len(sorted_fix) >= 4:
            detected_fix4 = sorted_fix[3]
            if len(detected_fix4) >= 5:
                fix4_button = check_fixation_on_button(detected_fix4[-2], detected_fix4[-1])

        # 5th longest fixation
        if len(sorted_fix) >= 5:
            detected_fix5 = sorted_fix[4]
            if len(detected_fix5) >= 5:
                fix5_button = check_fixation_on_button(detected_fix5[-2], detected_fix5[-1])

    except Exception as e:
        print(f"Warning: could not compute fixation buttons: {e}")

    return fix1_button, fix2_button, fix3_button, fix4_button, fix5_button


def fixation_durations_and_final_clicks(base_dir):
    """
    ...
    """
    subjects = [os.path.join(base_dir, s) for s in os.listdir(base_dir)]

    # Initialize counters by category
    categories = ["positive", "neutral", "negative"]
    correct_counts = {cat: {i: 0 for i in range(1, 6)} for cat in categories}
    incorrect_counts = {cat: 0 for cat in categories}
    total_counts = {cat: 0 for cat in categories}

    for s in subjects:
        try:
            data = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))
            ratings = pd.read_csv(os.path.join(s, "ratings.csv"), index_col="image_name")
        except FileNotFoundError as e:
            print(f"Warning: missing file for subject {s}: {e}")
            continue

        rating_segments = extract_emotion_rating_segments(data)

        for stim_id, emotions_dict in rating_segments.items():
            if stim_id not in ratings.index:
                print(f"Warning: {stim_id} missing in ratings for subject {s}")
                continue

            for emotion, df_segment in emotions_dict.items():
                try:
                    # Identify image category from stim_cat column
                    if "stim_cat" not in df_segment.columns:
                        print(f"Warning: stim_cat column missing in {stim_id} for subject {s}")
                        continue
                    stim_cat = str(df_segment["stim_cat"].iloc[0]).strip().lower()
                    if stim_cat not in categories:
                        print(f"Warning: unknown stim_cat '{stim_cat}' in {stim_id} for subject {s}")
                        continue

                    # Get top 5 fixations
                    fix_buttons = get_top_5_fixation_on_button(df_segment)
                    rating = int(ratings.loc[stim_id][emotion])

                    total_counts[stim_cat] += 1
                    matched = False

                    for i, fix in enumerate(fix_buttons, start=1):
                        if rating == fix:
                            correct_counts[stim_cat][i] += 1
                            matched = True
                            break

                    if not matched:
                        incorrect_counts[stim_cat] += 1

                except KeyError:
                    print(f"Warning: {emotion} not found for {stim_id} in ratings of {s}")
                except Exception as e:
                    print(f"Warning: error processing {stim_id}, {emotion} for {s}: {e}")

    # print summary
    for cat in categories:
        print(f"\nStimulus category: {cat.upper()}")
        print(f"Total number of rating trials processed: {total_counts[cat]}")
        for i in range(1, 6):
            print(f"Top {i} longest fixation duration, with matching button click: {correct_counts[cat][i]}")
        print(f"Incorrect (doesn't match any top 5 fixation durations): {incorrect_counts[cat]}")


def compute_fixation_metrics_0(base_dir, save_dir="fixation_metrics"):
    """
    Compute fixation metrics for stimulus viewing and emotion rating screens.
    Returns both subject-level and group-level summaries, and saves CSVs.
    """
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # ---- Stimulus viewing segments ----
        stim_segs = extract_stim_viewing_segments(df)
        for stim_id, seg_df in stim_segs.items():
            # Assuming emotion_category column exists (positive, neutral, negative)
            emotion_category = seg_df['emotion_category'].iloc[0] if 'emotion_category' in seg_df.columns else 'NA'
            
            x, y, t = convert_to_pygaze_compatible_format(seg_df)
            _, Efix = fixation_detection(x, y, t)
            
            durations = [f[2] for f in Efix] if Efix else []
            fixation_count = len(durations)
            mean_duration = np.mean(durations) if durations else 0
            total_duration = np.sum(durations) if durations else 0

            subject_summaries.append({
                'subject_id': subject_id,
                'segment_type': 'stimulus_viewing',
                'stim_id': stim_id,
                'emotion_category': emotion_category,
                'emotion': np.nan,
                'fixation_count': fixation_count,
                'mean_duration': mean_duration,
                'total_duration': total_duration
            })

        # ---- Emotion rating segments ----
        rating_segs = extract_emotion_rating_segments(df)
        for stim_id, emotions_dict in rating_segs.items():
            for emotion, seg_df in emotions_dict.items():
                x, y, t = convert_to_pygaze_compatible_format(seg_df)
                _, Efix = fixation_detection(x, y, t)
                
                durations = [f[2] for f in Efix] if Efix else []
                fixation_count = len(durations)
                mean_duration = np.mean(durations) if durations else 0
                total_duration = np.sum(durations) if durations else 0

                subject_summaries.append({
                    'subject_id': subject_id,
                    'segment_type': 'emotion_rating',
                    'stim_id': stim_id,
                    'emotion_category': np.nan,
                    'emotion': emotion,
                    'fixation_count': fixation_count,
                    'mean_duration': mean_duration,
                    'total_duration': total_duration
                })

    # ---- Convert to DataFrame ----
    subject_df = pd.DataFrame(subject_summaries)

    # ---- Group-level summaries ----
    # Stimulus viewing by emotion category
    stim_group = subject_df[subject_df['segment_type'] == 'stimulus_viewing'].groupby('emotion_category').agg(
        mean_fixation_count=('fixation_count', 'mean'),
        sd_fixation_count=('fixation_count', 'std'),
        mean_fixation_duration=('mean_duration', 'mean'),
        sd_fixation_duration=('mean_duration', 'std'),
        mean_total_duration=('total_duration', 'mean'),
        sd_total_duration=('total_duration', 'std')
    ).reset_index()

    # Emotion rating by emotion
    rating_group = subject_df[subject_df['segment_type'] == 'emotion_rating'].groupby('emotion').agg(
        mean_fixation_count=('fixation_count', 'mean'),
        sd_fixation_count=('fixation_count', 'std'),
        mean_fixation_duration=('mean_duration', 'mean'),
        sd_fixation_duration=('mean_duration', 'std'),
        mean_total_duration=('total_duration', 'mean'),
        sd_total_duration=('total_duration', 'std')
    ).reset_index()

    # ---- Save CSVs ----
    os.makedirs(save_dir, exist_ok=True)
    subject_csv_path = os.path.join(save_dir, "subject_level_fixation_metrics.csv")
    stim_group_csv_path = os.path.join(save_dir, "group_level_stimulus_viewing_metrics.csv")
    rating_group_csv_path = os.path.join(save_dir, "group_level_emotion_rating_metrics.csv")

    subject_df.to_csv(subject_csv_path, index=False)
    stim_group.to_csv(stim_group_csv_path, index=False)
    rating_group.to_csv(rating_group_csv_path, index=False)

    print(f"Saved subject-level metrics to: {subject_csv_path}")
    print(f"Saved group-level stimulus viewing metrics to: {stim_group_csv_path}")
    print(f"Saved group-level emotion rating metrics to: {rating_group_csv_path}")

    return subject_df, stim_group, rating_group


# emotion ratings grouped by stim_cat
def compute_fixation_metrics_collapsed(base_dir, save_dir="fixation_metrics_collapsed"):
    """
    Compute fixation metrics for emotion rating screens,
    collapsed within each emotion category (positive, neutral, negative).
    Produces both subject-level and group-level summaries.
    """

    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # Extract rating segments
        rating_segs = extract_emotion_rating_segments(df)

        # Container to accumulate by category
        category_accum = defaultdict(list)

        for stim_id, emotions_dict in rating_segs.items():
            for emotion, seg_df in emotions_dict.items():
                emotion_category = seg_df['stim_cat'].iloc[0]

                x, y, t = convert_to_pygaze_compatible_format(seg_df)
                _, Efix = fixation_detection(x, y, t)

                durations = [f[2] for f in Efix] if Efix else []
                if durations:
                    category_accum[emotion_category].extend(durations)

        # Compute mean per category
        for category, dur_list in category_accum.items():
            fixation_count = len(dur_list)
            mean_duration = np.mean(dur_list)
            total_duration = np.sum(dur_list)

            subject_summaries.append({
                'subject_id': subject_id,
                'emotion_category': category,
                'fixation_count': fixation_count,
                'mean_duration': mean_duration,
                'total_duration': total_duration
            })

    # ---- Convert to DataFrame ----
    subject_df = pd.DataFrame(subject_summaries)

    # ---- Group-level summaries ----
    group_df = subject_df.groupby('emotion_category').agg(
        mean_fixation_count=('fixation_count', 'mean'),
        sd_fixation_count=('fixation_count', 'std'),
        mean_fixation_duration=('mean_duration', 'mean'),
        sd_fixation_duration=('mean_duration', 'std'),
        mean_total_duration=('total_duration', 'mean'),
        sd_total_duration=('total_duration', 'std')
    ).reset_index()

    # ---- Save results ----
    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_level_fixation_metrics_collapsed.csv")
    group_path = os.path.join(save_dir, "group_level_fixation_metrics_collapsed.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


# metrics per emotion rating screen
def compute_fixation_metrics_per_rating(base_dir, save_dir="fixation_metrics_per_emotion"):
    """
    Compute fixation metrics for each emotion rating screen (fear, anger, happy, sad, disgust)
    within each stimulus for each subject.
    Produces both subject-level and group-level summaries.
    """
    
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # Extract rating segments (emotion-specific)
        rating_segs = extract_emotion_rating_segments(df)

        for stim_id, emotions_dict in rating_segs.items():
            for emotion, seg_df in emotions_dict.items():
                emotion_category = seg_df['stim_cat'].iloc[0]

                # Convert and compute fixations
                x, y, t = convert_to_pygaze_compatible_format(seg_df)
                _, Efix = fixation_detection(x, y, t)

                durations = [f[2] for f in Efix] if Efix else []
                fixation_count = len(durations)
                mean_duration = np.mean(durations) if durations else 0
                total_duration = np.sum(durations) if durations else 0

                subject_summaries.append({
                    'subject_id': subject_id,
                    'stim_id': stim_id,
                    'emotion_category': emotion_category,
                    'emotion': emotion,
                    'fixation_count': fixation_count,
                    'mean_duration': mean_duration,
                    'total_duration': total_duration
                })

    # ---- Convert to DataFrame ----
    subject_df = pd.DataFrame(subject_summaries)

    # ---- Group-level summaries ----
    group_df = subject_df.groupby(['emotion_category', 'emotion']).agg(
        mean_fixation_count=('fixation_count', 'mean'),
        sd_fixation_count=('fixation_count', 'std'),
        mean_fixation_duration=('mean_duration', 'mean'),
        sd_fixation_duration=('mean_duration', 'std'),
        mean_total_duration=('total_duration', 'mean'),
        sd_total_duration=('total_duration', 'std')
    ).reset_index()

    # ---- Save results ----
    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_level_fixation_metrics_per_emotion.csv")
    group_path = os.path.join(save_dir, "group_level_fixation_metrics_per_emotion.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df
