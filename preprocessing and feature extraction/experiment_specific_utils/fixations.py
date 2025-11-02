import os, re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from experiment_specific_utils.common import extract_stim_viewing_segments, extract_emotion_rating_segments
from extract_eye_features.fixations import convert_to_pygaze_compatible_format
from extract_eye_features.pygazeanalyser_fixations import fixation_detection


def compute_fixation_metrics_viewing(base_dir, save_dir="fixation_metrics_viewing"):
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
    subject_path = os.path.join(save_dir, "subject_level_fixation_metrics_viewing.csv")
    group_path = os.path.join(save_dir, "group_level_fixation_metrics_viewing.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


def plot_fixation_metrics_viewing(df, save_dir="fixation_metrics_viewing_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # categories on x-axis
    categories = ["positive", "negative", "neutral"]
    metrics = [
        ("mean_fixation_count", "Mean fixation count"),
        ("mean_fixation_duration", "Mean fixation duration (ms)"),
        ("mean_total_duration", "Mean total fixation duration (ms)")
    ]

    # visual params
    box_width = 0.3
    jitter_rel = 0.06
    dot_size = 18
    alpha_box = 0.65
    alpha_points = 0.95
    signif_alpha = 0.05

    # === I/O & PREP ===
    df.columns = df.columns.str.strip()
    if "subject_id" not in df.columns:
        raise RuntimeError("CSV must contain 'subject_id' column.")
    if "stimulus_category" not in df.columns:
        raise RuntimeError("CSV must contain 'stimulus_category' column.")

    df["stimulus_category"] = df["stimulus_category"].astype(str).str.strip().str.lower()

    # colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    category_colors = {cat: color_cycle[i % len(color_cycle)] for i, cat in enumerate(categories)}

    def sanitize_filename(s):
        return re.sub(r"[^\w\-_. ]", "_", s).replace(" ", "_")

    # positions and jitter
    positions = np.arange(len(categories))
    jitter_sigma = jitter_rel * max(1.0, len(categories))

    # === MAIN: one plot per metric ===
    pair_list = [("positive", "negative"), ("positive", "neutral"), ("negative", "neutral")]

    for metric_col, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))

        # prepare data per category for boxplots
        data_for_boxes = []
        cats_present = []
        pos_map = {}
        for i, cat in enumerate(categories):
            vals = df.loc[df["stimulus_category"] == cat, metric_col].dropna().values
            if vals.size > 0:
                data_for_boxes.append(vals)
                cats_present.append(cat)
                pos_map[cat] = i  # map category to x index
            else:
                # keep placeholders for consistent x positions
                data_for_boxes.append(np.array([]))
                cats_present.append(cat)
                pos_map[cat] = i

        # draw boxplots: use only non-empty arrays to avoid matplotlib warnings about empty array
        bp_positions = [pos for pos, arr in zip(positions, data_for_boxes)]
        # To ensure consistent ordering keep positions for all categories but supply empty arrays where needed
        bp = ax.boxplot(
            [arr if arr.size > 0 else [np.nan] for arr in data_for_boxes],
            positions=bp_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False
        )
        # apply colors, but skip NaN-only boxes by still coloring them (transparent)
        for i, patch in enumerate(bp["boxes"]):
            col = category_colors[categories[i]]
            patch.set_facecolor(col)
            patch.set_alpha(alpha_box)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        for whisker in bp.get("whiskers", []):
            whisker.set_color("black"); whisker.set_linewidth(0.8)
        for cap in bp.get("caps", []):
            cap.set_color("black"); cap.set_linewidth(0.8)

        # overlay subject points with jitter
        for _, row in df.iterrows():
            cat = str(row["stimulus_category"]).lower()
            if cat not in categories:
                continue
            val = row.get(metric_col, np.nan)
            if pd.isna(val):
                continue
            base_x = pos_map[cat]
            x_j = base_x + np.random.normal(0, jitter_sigma)
            ax.scatter(x_j, val, color=category_colors[cat], s=dot_size,
                    edgecolor="k", linewidth=0.25, alpha=alpha_points, zorder=6)

        # STATISTICAL TESTS: paired t-tests between category pairs across subjects
        pvals = []
        tested_pairs = []
        pair_results = []
        for a, b in pair_list:
            A = df.loc[df["stimulus_category"] == a, ["subject_id", metric_col]].dropna()
            B = df.loc[df["stimulus_category"] == b, ["subject_id", metric_col]].dropna()
            if A.empty or B.empty:
                pair_results.append((a, b, None, None, 0))
                continue
            merged = pd.merge(A, B, on="subject_id", suffixes=("_a", "_b"))
            n_paired = merged.shape[0]
            if n_paired < 2:
                pair_results.append((a, b, None, None, n_paired))
                continue
            vals_a = merged[f"{metric_col}_a"].values
            vals_b = merged[f"{metric_col}_b"].values
            tstat, pval = ttest_rel(vals_a, vals_b, nan_policy="omit")
            pvals.append(pval)
            tested_pairs.append((a, b, n_paired))
            pair_results.append((a, b, pval, (vals_a, vals_b), n_paired))

        if len(pvals) > 0:
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=signif_alpha, method="holm")
        else:
            reject, pvals_corrected = [], []

        # Annotate significant results
        # baseline y from data
        all_vals = df[metric_col].dropna().values
        if all_vals.size > 0:
            y_base = np.nanmax(all_vals)
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0] if (ylim[1] - ylim[0]) > 0 else 1.0
            start_offset = 0.04 * yrange
            stack_step = 0.06 * yrange
            used = 0
            ci = 0
            for (a, b, pval, vals_pair, n_paired) in pair_results:
                if pval is None:
                    continue
                try:
                    idx = tested_pairs.index((a, b, n_paired))
                except ValueError:
                    idx = ci
                adj_p = pvals_corrected[idx]
                is_signif = adj_p < signif_alpha
                if is_signif:
                    x1 = pos_map[a]
                    x2 = pos_map[b]
                    current_y = y_base + start_offset + used * stack_step
                    padding = 0.01 * yrange
                    ax.plot([x1, x1, x2, x2], [current_y, current_y + padding, current_y + padding, current_y], color="k", linewidth=1.0)
                    if adj_p < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {adj_p:.3f}"
                    ax.text((x1 + x2) / 2.0, current_y + padding + 0.005 * yrange, p_text,
                            ha="center", va="bottom", fontsize=9, color="k")
                    used += 1
                ci += 1

        # aesthetics
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([c.capitalize() for c in categories])
        legend_handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=category_colors[c],
                markersize=8, label=c.capitalize()) for c in categories
        ]
        ax.legend(handles=legend_handles, title="Stimulus category", loc="upper right")

        plt.tight_layout()
        out_fname = sanitize_filename(metric_col) + ".png"
        out_path = os.path.join(save_dir, out_fname)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("Saved:", out_path)


def compute_fixation_metrics_rating(base_dir, save_dir="fixation_metrics_rating"):
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # --- Clean columns ---
        df.columns = df.columns.str.strip()
        df['stim_cat'] = df['stim_cat'].astype(str).str.strip().replace({'nan': np.nan})

        # Extract rating segments
        rating_segs = extract_emotion_rating_segments(df)

        # Container: store metrics per stimulus category × emotion rating
        subj_cat_emotion_metrics = defaultdict(lambda: defaultdict(list))

        for stim_id, emotions_dict in rating_segs.items():
            for emotion_rating, seg_df in emotions_dict.items():

                # Assign stimulus category
                stim_cat = seg_df['stim_cat'].iloc[0]
                if pd.isna(stim_cat) or str(stim_cat).strip() == "":
                    fallback = df.loc[df['stim_id'] == stim_id, 'stim_cat']
                    if not fallback.empty and fallback.dropna().any():
                        stim_cat = fallback.ffill().bfill().dropna().iloc[0]
                    else:
                        stim_cat = "UNKNOWN"
                stim_cat = str(stim_cat).strip()

                # --- Calculate fixations for this segment ---
                x, y, t = convert_to_pygaze_compatible_format(seg_df)
                _, Efix = fixation_detection(x, y, t)

                durations = [f[2] for f in Efix] if Efix else []

                if durations:
                    subj_cat_emotion_metrics[stim_cat][emotion_rating].append({
                        'fixation_count': len(durations),
                        'total_duration': np.sum(durations),
                        'mean_duration': np.mean(durations)
                    })

        # --- Aggregate metrics per stimulus category × emotion rating ---
        for stim_cat, emotion_dict in subj_cat_emotion_metrics.items():
            for emotion_rating, metrics_list in emotion_dict.items():
                fixation_counts = np.array([m['fixation_count'] for m in metrics_list])
                total_durations = np.array([m['total_duration'] for m in metrics_list])
                mean_durations = np.array([m['mean_duration'] for m in metrics_list])

                subject_summaries.append({
                    'subject_id': subject_id,
                    'stimulus_category': stim_cat,
                    'emotion_rating': emotion_rating,
                    'mean_fixation_count': fixation_counts.mean(),
                    'sd_fixation_count': fixation_counts.std(ddof=0),
                    'mean_fixation_duration': mean_durations.mean(),
                    'sd_fixation_duration': mean_durations.std(ddof=0),
                    'mean_total_duration': total_durations.mean(),
                    'sd_total_duration': total_durations.std(ddof=0)
                })

    # ---- Convert to DataFrame ----
    subject_df = pd.DataFrame(subject_summaries)

    # ---- Group-level summaries ----
    group_df = (
        subject_df.groupby(['stimulus_category', 'emotion_rating'])
        .agg(
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
    subject_path = os.path.join(save_dir, "subject_level_fixation_metrics.csv")
    group_path = os.path.join(save_dir, "group_level_fixation_metrics.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


def plot_fixation_metrics_rating(df, out_dir="fixation_metrics_rating_plots"):
    os.makedirs(out_dir, exist_ok=True)

    emotions = ["happy", "sad", "anger", "disgust", "fear"]
    categories = ["positive", "negative", "neutral"]
    metrics = [
        ("mean_fixation_count", "Mean fixation count"),
        ("mean_fixation_duration", "Mean fixation duration (ms)"),
        ("mean_total_duration", "Mean total fixation duration (ms)")
    ]

    # Visual spacing and sizing controls
    group_spacing = 2.6        # distance between emotion group centers
    within_group_span = 1.0    # total horizontal span for boxes within a group
    box_width = 0.30
    jitter_rel = 0.06
    dot_size = 18
    alpha_box = 0.65
    alpha_points = 0.95
    signif_alpha = 0.05        # familywise alpha used after Holm correction

    # === I/O & PREP ===
    df.columns = df.columns.str.strip()
    # ensure subject_id exists
    if "subject_id" not in df.columns:
        raise RuntimeError("CSV must contain 'subject_id' column.")
    df["stimulus_category"] = df["stimulus_category"].astype(str).str.strip().str.lower()
    df["emotion_rating"] = df["emotion_rating"].astype(str).str.strip().str.lower()

    # compute category offsets within a group
    n_cat = len(categories)
    offsets = np.linspace(-within_group_span / 2.0, within_group_span / 2.0, n_cat)
    category_to_offset = {cat: off for cat, off in zip(categories, offsets)}

    # colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    category_colors = {cat: color_cycle[i % len(color_cycle)] for i, cat in enumerate(categories)}

    def sanitize_filename(s):
        return re.sub(r"[^\w\-_. ]", "_", s).replace(" ", "_")

    # jitter scaled to group spacing
    jitter_sigma = jitter_rel * group_spacing

    # figure width scaled to number of groups
    fig_width = max(10, group_spacing * len(emotions) * 1.1)

    # helper: compute x position for a given emotion index and category
    def x_pos_for(emotion_idx, category):
        return emotion_idx * group_spacing + category_to_offset[category]

    # === MAIN: one plot per metric with stats annotations ===
    for metric_col, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # prepare data for boxplots
        data_for_boxes = []
        positions = []
        box_colors = []
        pos_meta = []

        for e_idx, emo in enumerate(emotions):
            for cat in categories:
                vals = df.loc[
                    (df["emotion_rating"] == emo) & (df["stimulus_category"] == cat),
                    [ "subject_id", metric_col ]
                ].dropna(subset=[metric_col])
                if vals.shape[0] > 0:
                    data_for_boxes.append(vals[metric_col].values)
                    pos = x_pos_for(e_idx, cat)
                    positions.append(pos)
                    box_colors.append(category_colors[cat])
                    pos_meta.append((emo, cat))

        # draw boxplots
        if data_for_boxes:
            bp = ax.boxplot(
                data_for_boxes,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha_box)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)
            for whisker in bp.get("whiskers", []):
                whisker.set_color("black"); whisker.set_linewidth(0.8)
            for cap in bp.get("caps", []):
                cap.set_color("black"); cap.set_linewidth(0.8)

        # overlay subject points with jitter
        for _, row in df.iterrows():
            emo = str(row["emotion_rating"]).lower()
            cat = str(row["stimulus_category"]).lower()
            if emo not in emotions or cat not in categories:
                continue
            val = row.get(metric_col, np.nan)
            if pd.isna(val):
                continue
            e_idx = emotions.index(emo)
            base_x = x_pos_for(e_idx, cat)
            x_j = base_x + np.random.normal(0, jitter_sigma)
            ax.scatter(x_j, val, color=category_colors[cat], s=dot_size,
                    edgecolor="k", linewidth=0.25, alpha=alpha_points, zorder=6)

        # STATISTICAL TESTS: paired t-tests within each emotion for the 3 category pairs
        pair_list = [("positive", "negative"), ("positive", "neutral"), ("negative", "neutral")]

        for e_idx, emo in enumerate(emotions):
            # collect p-values and record which pairs were tested
            pvals = []
            tested_pairs = []
            pair_results = []

            for a, b in pair_list:
                # get matched subject measurements for this emotion for categories a and b
                A = df.loc[(df["emotion_rating"] == emo) & (df["stimulus_category"] == a), ["subject_id", metric_col]].dropna()
                B = df.loc[(df["emotion_rating"] == emo) & (df["stimulus_category"] == b), ["subject_id", metric_col]].dropna()
                if A.empty or B.empty:
                    # cannot test if missing one group
                    pair_results.append((a, b, None, None, 0))
                    continue
                merged = pd.merge(A, B, on="subject_id", suffixes=("_a", "_b"))
                n_paired = merged.shape[0]
                if n_paired < 2:
                    pair_results.append((a, b, None, None, n_paired))
                    continue
                vals_a = merged[f"{metric_col}_a"].values
                vals_b = merged[f"{metric_col}_b"].values
                # paired t-test
                tstat, pval = ttest_rel(vals_a, vals_b, nan_policy="omit")
                pvals.append(pval)
                tested_pairs.append((a, b, n_paired))
                pair_results.append((a, b, pval, (vals_a, vals_b), n_paired))

            # apply Holm correction on pvals if any
            if len(pvals) > 0:
                reject, pvals_corrected, _, _ = multipletests(pvals, alpha=signif_alpha, method="holm")
            else:
                reject, pvals_corrected = [], []

            # Annotate significant results
            # we need a stacking offset so multiple comparisons in same group don't overlap
            # compute an initial y baseline using existing data values for this emotion
            emo_vals = df.loc[df["emotion_rating"] == emo, metric_col].dropna().values
            if emo_vals.size == 0:
                continue
            y_base = np.nanmax(emo_vals)
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0] if (ylim[1] - ylim[0]) > 0 else 1.0
            # start a little above the max and stack upward
            start_offset = 0.04 * yrange
            stack_step = 0.06 * yrange
            used = 0

            # iterate pair_results in same order as pvals list to map corrected p-values
            ci = 0
            for (a, b, pval, vals_pair, n_paired) in pair_results:
                if pval is None:
                    continue
                # find index in tested_pairs to map to corrected pvals
                # tested_pairs stores (a,b,n_paired) in same order as pvals
                # locate matching entry
                try:
                    idx = tested_pairs.index((a, b, n_paired))
                except ValueError:
                    # fallback: iterate mapping by insertion order
                    idx = ci
                adj_p = pvals_corrected[idx]
                is_signif = False
                if adj_p < signif_alpha:
                    is_signif = True

                if is_signif:
                    # compute x positions for boxes a and b for this emotion
                    x1 = x_pos_for(e_idx, a)
                    x2 = x_pos_for(e_idx, b)
                    # compute y coordinate to draw line
                    current_y = y_base + start_offset + used * stack_step
                    # draw bracket line
                    padding = 0.01 * yrange
                    ax.plot([x1, x1, x2, x2], [current_y, current_y + padding, current_y + padding, current_y], color="k", linewidth=1.0)
                    # format p-value text
                    if adj_p < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {adj_p:.3f}"
                    # place text centered
                    ax.text((x1 + x2) / 2.0, current_y + padding + 0.005 * yrange, p_text,
                            ha="center", va="bottom", fontsize=9, color="k")
                    used += 1
                ci += 1

        # aesthetics
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)
        centers = [i * group_spacing for i in range(len(emotions))]
        ax.set_xticks(centers)
        ax.set_xticklabels([e.capitalize() for e in emotions])
        legend_handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=category_colors[c],
                markersize=8, label=c.capitalize()) for c in categories
        ]
        ax.legend(handles=legend_handles, title="Stimulus category", loc="upper right")

        plt.tight_layout()
        out_fname = sanitize_filename(metric_col) + ".png"
        out_path = os.path.join(out_dir, out_fname)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("Saved:", out_path)


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
