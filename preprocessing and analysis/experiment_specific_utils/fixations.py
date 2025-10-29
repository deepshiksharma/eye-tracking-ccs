import os
import pandas as pd
import numpy as np
from collections import defaultdict
from experiment_specific_utils.common import extract_stim_viewing_segments, extract_emotion_rating_segments
from compute_eye_metrics.fixations import convert_to_pygaze_compatible_format
from compute_eye_metrics.pygazeanalyser_fixations import fixation_detection


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

def get_top_7_fixation_on_button(dataframe):
    """
    Return a 7-tuple of button IDs corresponding to the top-7 longest fixations.
    Uses helpers expected to exist in the calling environment:
      - convert_to_pygaze_compatible_format(df) -> (x, y, t)
      - fixation_detection(x, y, t) -> (Sfix, Efix) where each Efix entry is
        [starttime, endtime, duration, endx, endy]
      - check_fixation_on_button(x, y) -> integer button id (or 0 for none)
    Robust to empty Efix and exceptions. Always returns 7 integers.
    """
    defaults = [0] * 7
    try:
        x, y, t = convert_to_pygaze_compatible_format(dataframe)
        Sfix, Efix = fixation_detection(x, y, t)
        if not Efix:
            return tuple(defaults)

        # Sort fixations by duration (index 2) descending
        sorted_fix = sorted(Efix, key=lambda i: i[2] if len(i) > 2 else 0, reverse=True)

        results = []
        for i in range(7):
            if i < len(sorted_fix) and len(sorted_fix[i]) >= 5:
                endx = sorted_fix[i][-2]
                endy = sorted_fix[i][-1]
                try:
                    btn = check_fixation_on_button(endx, endy) or 0
                except Exception:
                    btn = 0
                results.append(int(btn))
            else:
                results.append(0)

        return tuple(results)

    except Exception as e:
        print(f"Warning: could not compute fixation buttons: {e}")
        return tuple(defaults)

def fixation_durations_and_final_clicks_top_7(base_dir):
    """
    Walk subjects in base_dir, extract top-7 fixation buttons for each rating segment,
    and print counts of matches between rating value and top-N fixation-on-button IDs.
    Expected helpers in environment:
      - extract_emotion_rating_segments(data) -> dict: {stim_id: {emotion: df_segment, ...}, ...}
    """
    # build subject paths (only directories)
    subjects = [os.path.join(base_dir, s) for s in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, s))]

    categories = ["positive", "neutral", "negative"]
    # initialize counters for top 1..7
    correct_counts = {cat: {i: 0 for i in range(1, 8)} for cat in categories}
    incorrect_counts = {cat: 0 for cat in categories}
    total_counts = {cat: 0 for cat in categories}

    for s in subjects:
        try:
            data = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"), low_memory=False)
            ratings = pd.read_csv(os.path.join(s, "ratings.csv"), index_col="image_name")
        except FileNotFoundError as e:
            print(f"Warning: missing file for subject {s}: {e}")
            continue
        except Exception as e:
            print(f"Warning: error reading files for subject {s}: {e}")
            continue

        rating_segments = extract_emotion_rating_segments(data)

        for stim_id, emotions_dict in rating_segments.items():
            if stim_id not in ratings.index:
                print(f"Warning: {stim_id} missing in ratings for subject {s}")
                continue

            for emotion, df_segment in emotions_dict.items():
                try:
                    # Validate category column
                    if "stim_cat" not in df_segment.columns:
                        print(f"Warning: stim_cat column missing in {stim_id} for subject {s}")
                        continue
                    stim_cat = str(df_segment["stim_cat"].iloc[0]).strip().lower()
                    if stim_cat not in categories:
                        print(f"Warning: unknown stim_cat '{stim_cat}' in {stim_id} for subject {s}")
                        continue

                    # Get top 7 fixation-on-button IDs
                    fix_buttons = get_top_7_fixation_on_button(df_segment)
                    if not isinstance(fix_buttons, (list, tuple)) or len(fix_buttons) < 1:
                        fix_buttons = tuple([0]*7)
                    # Ensure length exactly 7
                    if len(fix_buttons) != 7:
                        fix_buttons = tuple(list(fix_buttons)[:7] + [0] * max(0, 7 - len(fix_buttons)))

                    # Safely get rating value
                    try:
                        rating_val = ratings.at[stim_id, emotion]
                        # if rating is NaN or empty, treat as mismatch
                        if pd.isna(rating_val):
                            raise ValueError("NaN rating")
                        rating = int(rating_val)
                    except KeyError:
                        print(f"Warning: {emotion} not found for {stim_id} in ratings of {s}")
                        continue
                    except Exception:
                        print(f"Warning: invalid rating for {stim_id}, {emotion} in {s}")
                        continue

                    total_counts[stim_cat] += 1
                    matched = False

                    for i, fix in enumerate(fix_buttons, start=1):
                        if rating == fix:
                            # Increment appropriate counter for top-i
                            if i in correct_counts[stim_cat]:
                                correct_counts[stim_cat][i] += 1
                            else:
                                correct_counts[stim_cat][i] = 1
                            matched = True
                            break

                    if not matched:
                        incorrect_counts[stim_cat] += 1

                except Exception as e:
                    print(f"Warning: error processing {stim_id}, {emotion} for {s}: {e}")

    # print summary
    for cat in categories:
        print(f"\nStimulus category: {cat.upper()}")
        print(f"Total number of rating trials processed: {total_counts[cat]}")
        for i in range(1, 8):
            print(f"Top {i} longest fixation duration, with matching button click: {correct_counts[cat].get(i,0)}")
        print(f"Incorrect (doesn't match any top 7 fixation durations): {incorrect_counts[cat]}")


def compute_fixation_metrics_during_viewing(base_dir, save_dir="fixation_metrics_during_viewing"):
    """
    Compute fixation metrics per stimulus viewing segment, grouped by stimulus category.
    Each row in the subject-level CSV corresponds to one subject × stimulus_category.
    Produces both subject-level and group-level summaries.

    Assumes the following helper functions exist in scope:
      - extract_stim_viewing_segments(dataframe) -> dict[stim_id] = dataframe_segment
      - convert_to_pygaze_compatible_format(df) -> (x, y, t)
      - fixation_detection(x, y, t) -> (Epos, Efix) where Efix is iterable of fixations and
        each fixation has duration at index 2 (same as original code).
    """
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # --- Clean columns ---
        df.columns = df.columns.str.strip()
        df['stim_cat'] = df['stim_cat'].astype(str).str.strip().replace({'nan': np.nan})

        # Extract viewing segments (stim_present == True)
        viewing_segs = extract_stim_viewing_segments(df)
        # Container: store metrics per stimulus category
        subj_cat_metrics = defaultdict(list)

        for stim_id, seg_df in viewing_segs.items():
            if seg_df.empty:
                continue

            # Assign stimulus category with fallback if missing
            stim_cat = seg_df.get('stim_cat', pd.Series([np.nan])).iloc[0]
            if pd.isna(stim_cat) or str(stim_cat).strip() == "":
                fallback = df.loc[df['stim_id'] == stim_id, 'stim_cat']
                if not fallback.empty and fallback.dropna().any():
                    stim_cat = fallback.ffill().bfill().dropna().iloc[0]
                else:
                    stim_cat = "UNKNOWN"
            stim_cat = str(stim_cat).strip()

            # --- Calculate fixations for this viewing segment ---
            x, y, t = convert_to_pygaze_compatible_format(seg_df)
            _, Efix = fixation_detection(x, y, t)

            durations = [f[2] for f in Efix] if Efix else []

            if durations:
                subj_cat_metrics[stim_cat].append({
                    'fixation_count': len(durations),
                    'total_duration': np.sum(durations),
                    'mean_duration': np.mean(durations)
                })
            else:
                # If no fixations detected, record zeros to reflect viewing without fixations
                subj_cat_metrics[stim_cat].append({
                    'fixation_count': 0,
                    'total_duration': 0.0,
                    'mean_duration': 0.0
                })

        # --- Aggregate metrics per stimulus category for this subject ---
        for stim_cat, metrics_list in subj_cat_metrics.items():
            fixation_counts = np.array([m['fixation_count'] for m in metrics_list])
            total_durations = np.array([m['total_duration'] for m in metrics_list])
            mean_durations = np.array([m['mean_duration'] for m in metrics_list])

            subject_summaries.append({
                'subject_id': subject_id,
                'stimulus_category': stim_cat,
                'n_viewings': len(metrics_list),
                'mean_fixation_count': fixation_counts.mean() if fixation_counts.size > 0 else 0.0,
                'sd_fixation_count': fixation_counts.std(ddof=0) if fixation_counts.size > 0 else 0.0,
                'mean_fixation_duration': mean_durations.mean() if mean_durations.size > 0 else 0.0,
                'sd_fixation_duration': mean_durations.std(ddof=0) if mean_durations.size > 0 else 0.0,
                'mean_total_duration': total_durations.mean() if total_durations.size > 0 else 0.0,
                'sd_total_duration': total_durations.std(ddof=0) if total_durations.size > 0 else 0.0
            })

    # ---- Convert to DataFrame ----
    subject_df = pd.DataFrame(subject_summaries)

    # If no data then create empty group_df with expected columns
    if subject_df.empty:
        group_df = pd.DataFrame(columns=[
            'stimulus_category',
            'n_subjects', 
            'mean_fixation_count',
            'sd_fixation_count',
            'mean_fixation_duration',
            'sd_fixation_duration',
            'mean_total_duration',
            'sd_total_duration'
        ])
    else:
        # ---- Group-level summaries ----
        # Aggregate across subjects for each stimulus_category
        group_df = (
            subject_df.groupby(['stimulus_category'])
            .agg(
                n_subjects=('subject_id', 'nunique'),
                mean_fixation_count=('mean_fixation_count', 'mean'),
                sd_fixation_count=('mean_fixation_count', 'std'),
                mean_fixation_duration=('mean_fixation_duration', 'mean'),
                sd_fixation_duration=('mean_fixation_duration', 'std'),
                mean_total_duration=('mean_total_duration', 'mean'),
                sd_total_duration=('mean_total_duration', 'std')
            )
            .reset_index()
        )

    # ---- Save CSVs ----
    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_lvl_fix_viewing.csv")
    group_path = os.path.join(save_dir, "group_lvl_fix_viewing.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df
    # subject_viewing_fixations, group_viewing_fixations = compute_fixation_metrics_per_viewing_segment(base_dir)
