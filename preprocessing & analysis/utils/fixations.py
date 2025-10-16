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



