import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from .common import remove_invalids
from .pupildiameter import apply_smoothing, baseline_correction
from .fixations import convert_to_pygaze_compatible_format
from .pygazeanalyser_fixations import fixation_detection


# _____________________________________________
# EXPERIMENT SPECIFIC, GENERALLY USED FUNCTIONS
# _____________________________________________

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

# ______________________________________________________
# EXPERIMENT SPECIFIC, FUNCTIONS USED FOR PUPIL DIAMETER
# ______________________________________________________

def compute_mean_sd(base_dir, save_csv=True, plot=True):
    """
    
    """

    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []
    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # clean, smooth, and baseline correct
        df = remove_invalids(df)
        df = apply_smoothing(df)
        df = baseline_correction(df)

        # keep only the rows where stim_present==True
        stim_df = df[df["stim_present"] == True].copy()
        if stim_df.empty:
            continue
        
        # ensure columns exist
        for col in ["left_pupil_diameter_smooth", "right_pupil_diameter_smooth",
                    "left_pupil_diameter_bc", "right_pupil_diameter_bc"]:
            if col not in stim_df.columns:
                stim_df[col] = np.nan

        # group by stim_cat and compute mean/sd separately for absolute and baseline corrected pupil diameters
        agg_abs = (
            stim_df.groupby("stim_cat")[["left_pupil_diameter_smooth", "right_pupil_diameter_smooth"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg_bc = (
            stim_df.groupby("stim_cat")[["left_pupil_diameter_bc", "right_pupil_diameter_bc"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        # merge the two aggregations on stim_cat
        agg = pd.merge(agg_abs, agg_bc, on="stim_cat", how="outer", suffixes=("_abs", "_bc"))

        # flatten multi-index columns if needed
        agg.columns = [
            "stim_cat",
            "left_mean_abs", "left_sd_abs",
            "right_mean_abs", "right_sd_abs",
            "left_mean_bc", "left_sd_bc",
            "right_mean_bc", "right_sd_bc",
        ]
        
        agg["subject_id"] = subject_id
        # reorder columns
        cols = ["subject_id", "stim_cat",
                "left_mean_abs", "left_sd_abs", "right_mean_abs", "right_sd_abs",
                "left_mean_bc", "left_sd_bc", "right_mean_bc", "right_sd_bc"]
        agg = agg[cols]

        subject_summaries.append(agg)

    # combine subject summaries
    if not subject_summaries:
        raise RuntimeError("No subject summaries were produced. Check input files and preprocessing functions.")

    subject_lvl = pd.concat(subject_summaries, ignore_index=True)

    # group-level summary across subjects: for each stim_cat, compute mean sd of subject-level means and sds
    group_lvl = (
        subject_lvl.groupby("stim_cat")
        .agg({
            "left_mean_abs": ["mean", "std"],
            "left_sd_abs": ["mean", "std"],
            "right_mean_abs": ["mean", "std"],
            "right_sd_abs": ["mean", "std"],
            "left_mean_bc": ["mean", "std"],
            "left_sd_bc": ["mean", "std"],
            "right_mean_bc": ["mean", "std"],
            "right_sd_bc": ["mean", "std"],
        })
        .reset_index()
    )


    # flatten multiindex columns
    group_lvl.columns = [
        "stim_cat",
        "left_mean_abs_mean",   "left_mean_abs_sd",
        "left_sd_abs_mean",     "left_sd_abs_sd",
        "right_mean_abs_mean",  "right_mean_abs_sd",
        "right_sd_abs_mean",    "right_sd_abs_sd",
        "left_mean_bc_mean",    "left_mean_bc_sd",
        "left_sd_bc_mean",      "left_sd_bc_sd",
        "right_mean_bc_mean",   "right_mean_bc_sd",
        "right_sd_bc_mean",     "right_sd_bc_sd",
    ]

    # clean up column names for readability
    rename_map = {
        "left_mean_abs_mean":   "left_ABS_mean",    # mean
        "left_mean_abs_sd":     "left_ABS_mean_sd", # sd of the mean
        "left_sd_abs_mean":     "left_ABS_sd_mean", # mean of the sd
        "left_sd_abs_sd":       "left_ABS_sd_sd",   # sd of the sd
        "right_mean_abs_mean":  "right_ABS_mean",
        "right_mean_abs_sd":    "right_ABS_mean_sd",
        "right_sd_abs_mean":    "right_ABS_sd_mean",
        "right_sd_abs_sd":      "right_ABS_sd_sd",
        "left_mean_bc_mean":    "left_BC_mean",
        "left_mean_bc_sd":      "left_BC_mean_sd",
        "left_sd_bc_mean":      "left_BC_sd_mean",
        "left_sd_bc_sd":        "left_BC_sd_sd",
        "right_mean_bc_mean":   "right_BC_mean",
        "right_mean_bc_sd":     "right_BC_mean_sd",
        "right_sd_bc_mean":     "right_BC_sd_mean",
        "right_sd_bc_sd":       "right_BC_sd_sd",
    }
    group_lvl.rename(columns=rename_map, inplace=True)
    
    if save_csv:
        group_lvl.to_csv("group_lvl_pupildia_summary.csv", index=False)
        subject_lvl.to_csv("subject_lvl_pupildia_summary.csv", index=False)
        print("Saved group and subject level summaries to csv.")

    if plot:
        categories = group_lvl['stim_cat'].tolist()
        x = np.arange(len(categories))  # x positions
        width = 0.35    # width of bars

        # plot absolute pupil diameters
        fig, ax = plt.subplots(figsize=(8, 5))
        left_means = group_lvl['left_ABS_mean']
        right_means = group_lvl['right_ABS_mean']
        left_sd = group_lvl['left_ABS_mean_sd']
        right_sd = group_lvl['right_ABS_mean_sd']
        bars_left = ax.bar(x - width/2, left_means, width, yerr=left_sd, capsize=5, label='Left eye')
        bars_right = ax.bar(x + width/2, right_means, width, yerr=right_sd, capsize=5, label='Right eye')
        # annotate bars
        for bar, mean, std in zip(bars_left, left_means, left_sd):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean:.2f}±{std:.2f}",
                    ha='center', va='bottom', fontsize=8)
        for bar, mean, std in zip(bars_right, right_means, right_sd):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean:.2f}±{std:.2f}",
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Pupil diameter (mm)')
        ax.set_title('Absolute pupil diameter per emotion category')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # plot baseline-corrected pupil diameters
        fig, ax = plt.subplots(figsize=(8,5))
        left_means_bc = group_lvl['left_BC_mean']
        right_means_bc = group_lvl['right_BC_mean']
        left_sd_bc = group_lvl['left_BC_mean_sd']
        right_sd_bc = group_lvl['right_BC_mean_sd']
        bars_left_bc = ax.bar(x - width/2, left_means_bc, width, yerr=left_sd_bc, capsize=5, label='Left Eye')
        bars_right_bc = ax.bar(x + width/2, right_means_bc, width, yerr=right_sd_bc, capsize=5, label='Right Eye')
        # annotate bars
        for bar, mean, std in zip(bars_left_bc, left_means_bc, left_sd_bc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean:.2f}±{std:.2f}",
                    ha='center', va='bottom', fontsize=8)
        for bar, mean, std in zip(bars_right_bc, right_means_bc, right_sd_bc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean:.2f}±{std:.2f}",
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Pupil diameter (mm)')
        ax.set_title('Baseline-corrected pupil diameter per emotion category')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return group_lvl, subject_lvl


def run_ttests(subject_lvl, save_txt=True):
    """
    
    """
    df = subject_lvl.copy()

    left_bc = df.pivot(index='subject_id', columns='stim_cat', values='left_mean_bc')
    right_bc = df.pivot(index='subject_id', columns='stim_cat', values='right_mean_bc')

    # run tests for baseline-corrected left pupil
    t_pos_neg, p_pos_neg = stats.ttest_rel(left_bc['positive'], left_bc['negative'])
    t_pos_neu, p_pos_neu = stats.ttest_rel(left_bc['positive'], left_bc['neutral'])
    t_neg_neu, p_neg_neu = stats.ttest_rel(left_bc['negative'], left_bc['neutral'])
    print("Left pupil (baseline-corrected):")
    print(f"Positive vs Negative: t={t_pos_neg:.5f}, p={p_pos_neg:.5f}")
    print(f"Positive vs Neutral: t={t_pos_neu:.5f}, p={p_pos_neu:.5f}")
    print(f"Negative vs Neutral: t={t_neg_neu:.5f}, p={p_neg_neu:.5f}")

    # run tests for baseline-corrected right pupil
    t_pos_neg_r, p_pos_neg_r = stats.ttest_rel(right_bc['positive'], right_bc['negative'])
    t_pos_neu_r, p_pos_neu_r = stats.ttest_rel(right_bc['positive'], right_bc['neutral'])
    t_neg_neu_r, p_neg_neu_r = stats.ttest_rel(right_bc['negative'], right_bc['neutral'])
    print("\nRight pupil (baseline-corrected):")
    print(f"Positive vs Negative: t={t_pos_neg_r:.5f}, p={p_pos_neg_r:.5f}")
    print(f"Positive vs Neutral: t={t_pos_neu_r:.5f}, p={p_pos_neu_r:.5f}")
    print(f"Negative vs Neutral: t={t_neg_neu_r:.5f}, p={p_neg_neu_r:.5f}")
    
    # run tests comparing left vs right pupil within each emotion
    t_neg_lr, p_neg_lr = stats.ttest_rel(left_bc['negative'], right_bc['negative'])
    t_neu_lr, p_neu_lr = stats.ttest_rel(left_bc['neutral'], right_bc['neutral'])
    t_pos_lr, p_pos_lr = stats.ttest_rel(left_bc['positive'], right_bc['positive'])
    print("\nLeft vs Right pupil (baseline-corrected):")
    print(f"Negative: t={t_neg_lr:.5f}, p={p_neg_lr:.5f}")
    print(f"Neutral: t={t_neu_lr:.5f}, p={p_neu_lr:.5f}")
    print(f"Positive: t={t_pos_lr:.5f}, p={p_pos_lr:.5f}")

    if save_txt:
        with open("pupil_diameters_ttests.txt", "w") as f:
            f.write("Left pupil (baseline-corrected):\n")
            f.write(f"Positive vs Negative: t={t_pos_neg:.5f}, p={p_pos_neg:.5f}\n")
            f.write(f"Positive vs Neutral: t={t_pos_neu:.5f}, p={p_pos_neu:.5f}\n")
            f.write(f"Negative vs Neutral: t={t_neg_neu:.5f}, p={p_neg_neu:.5f}\n")
            f.write("\nRight pupil (baseline-corrected):\n")
            f.write(f"Positive vs Negative: t={t_pos_neg_r:.5f}, p={p_pos_neg_r:.5f}\n")
            f.write(f"Positive vs Neutral: t={t_pos_neu_r:.5f}, p={p_pos_neu_r:.5f}\n")
            f.write(f"Negative vs Neutral: t={t_neg_neu_r:.5f}, p={p_neg_neu_r:.5f}\n")
            f.write("\nLeft vs Right pupil (baseline-corrected):\n")
            f.write(f"Negative: t={t_neg_lr:.5f}, p={p_neg_lr:.5f}\n")
            f.write(f"Neutral: t={t_neu_lr:.5f}, p={p_neu_lr:.5f}\n")
            f.write(f"Positive: t={t_pos_lr:.5f}, p={p_pos_lr:.5f}\n")


# _________________________________________________
# EXPERIMENT SPECIFIC, FUNCTIONS USED FOR FIXATIONS
# _________________________________________________

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
