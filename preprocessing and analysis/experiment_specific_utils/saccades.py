import os, re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from compute_eye_metrics.utils import replace_invalids_w_nans
from compute_eye_metrics.saccades import detect_saccades
from experiment_specific_utils.common import extract_stim_viewing_segments, extract_emotion_rating_segments


def compute_saccadic_metrics_viewing(base_dir, save_dir="saccadic_metrics_viewing"):
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df_path = os.path.join(s, "eye_tracking_data.csv")
        if not os.path.exists(df_path):
            continue

        df = pd.read_csv(df_path)
        df.columns = df.columns.str.strip()
        df['stim_cat'] = df['stim_cat'].astype(str).str.strip().replace({'nan': np.nan})

        viewing_segs = extract_stim_viewing_segments(df)

        # container: stimulus_category -> list of per-stim metrics
        subj_cat_metrics = defaultdict(list)

        for stim_id, seg_df in viewing_segs.items():
            if seg_df is None or seg_df.empty:
                continue

            # resolve stimulus category with fallback
            stim_cat = None
            if 'stim_cat' in seg_df.columns:
                try:
                    stim_cat = seg_df['stim_cat'].iloc[0]
                except Exception:
                    stim_cat = None
            if pd.isna(stim_cat) or stim_cat is None or str(stim_cat).strip() == "":
                fallback = df.loc[df['stim_id'] == stim_id, 'stim_cat']
                if not fallback.empty and fallback.dropna().any():
                    stim_cat = fallback.ffill().bfill().dropna().astype(str).str.strip().iloc[0]
                else:
                    stim_cat = "UNKNOWN"
            stim_cat = str(stim_cat).strip()

            # replace invalid stretches with NaN
            seg_df_clean = replace_invalids_w_nans(seg_df, verbose=False)

            # detect saccades for this viewing segment
            try:
                sacc_df, G, meta = detect_saccades(
                    seg_df_clean,
                    ts_col='device_time_stamp',
                    ts_unit_microsec=True,
                    fs=1200.0,
                    smoothing_window_samples=7,
                    vel_threshold_k=6.0,
                    min_duration_ms=10.0,
                    max_duration_ms=150.0,
                    min_amplitude_norm=0.02,
                    max_amplitude_norm=1.2,
                    merge_gap_ms=5.0,
                    interp_limit_s=0.02
                )
            except Exception as e:
                # skip this segment on error
                print(f"Error detecting saccades for {subject_id} stim {stim_id}: {e}")
                continue

            # if no saccades detected, record zeros to reflect viewing without saccades
            if sacc_df is None or sacc_df.empty:
                subj_cat_metrics[stim_cat].append({
                    'saccade_count': 0,
                    'total_duration': 0.0,
                    'mean_duration': 0.0,
                    'mean_amplitude': 0.0,
                    'mean_peak_velocity': 0.0
                })
                continue

            # per-segment saccade metrics
            sacc_count = int(len(sacc_df))
            total_duration = float(sacc_df['duration_ms'].sum()) if 'duration_ms' in sacc_df.columns else np.nan
            mean_duration = float(sacc_df['duration_ms'].mean()) if 'duration_ms' in sacc_df.columns else np.nan
            mean_amplitude = float(sacc_df['amplitude_norm'].mean()) if 'amplitude_norm' in sacc_df.columns else np.nan
            # keep same column name as original detector output if available
            peak_col = 'peak_velocity_norm_per_s' if 'peak_velocity_norm_per_s' in sacc_df.columns else ('peak_velocity' if 'peak_velocity' in sacc_df.columns else None)
            mean_peak_velocity = float(sacc_df[peak_col].mean()) if peak_col and peak_col in sacc_df.columns else np.nan

            subj_cat_metrics[stim_cat].append({
                'saccade_count': sacc_count,
                'total_duration': total_duration,
                'mean_duration': mean_duration,
                'mean_amplitude': mean_amplitude,
                'mean_peak_velocity': mean_peak_velocity
            })

        # aggregate per stimulus_category for this subject
        for stim_cat, metrics_list in subj_cat_metrics.items():
            if not metrics_list:
                continue

            saccade_counts = np.array([m['saccade_count'] for m in metrics_list], dtype=float)
            total_durations = np.array([m['total_duration'] for m in metrics_list], dtype=float)
            mean_durations = np.array([m['mean_duration'] for m in metrics_list], dtype=float)
            mean_amplitudes = np.array([m['mean_amplitude'] for m in metrics_list], dtype=float)
            mean_peak_vels = np.array([m['mean_peak_velocity'] for m in metrics_list], dtype=float)

            subject_summaries.append({
                'subject_id': subject_id,
                'stimulus_category': stim_cat,
                'n_viewings': int(len(metrics_list)),
                'mean_saccade_count': float(np.nanmean(saccade_counts)),
                'sd_saccade_count': float(np.nanstd(saccade_counts, ddof=0)),
                'mean_saccade_duration_ms': float(np.nanmean(mean_durations)),
                'sd_saccade_duration_ms': float(np.nanstd(mean_durations, ddof=0)),
                'mean_total_saccade_duration_ms': float(np.nanmean(total_durations)),
                'sd_total_saccade_duration_ms': float(np.nanstd(total_durations, ddof=0)),
                'mean_amplitude_norm': float(np.nanmean(mean_amplitudes)),
                'sd_amplitude_norm': float(np.nanstd(mean_amplitudes, ddof=0)),
                'mean_peak_velocity': float(np.nanmean(mean_peak_vels)),
                'sd_peak_velocity': float(np.nanstd(mean_peak_vels, ddof=0))
            })

    # convert to DataFrame
    subject_df = pd.DataFrame(subject_summaries)

    # group-level summaries aggregated by stimulus_category
    if not subject_df.empty:
        group_df = (
            subject_df.groupby(['stimulus_category'])
            .agg(
                n_subjects=('subject_id', 'nunique'),
                mean_saccade_count=('mean_saccade_count', 'mean'),
                sd_saccade_count=('mean_saccade_count', 'std'),
                mean_saccade_duration_ms=('mean_saccade_duration_ms', 'mean'),
                sd_saccade_duration_ms=('mean_saccade_duration_ms', 'std'),
                mean_total_saccade_duration_ms=('mean_total_saccade_duration_ms', 'mean'),
                sd_total_saccade_duration_ms=('mean_total_saccade_duration_ms', 'std'),
                mean_amplitude_norm=('mean_amplitude_norm', 'mean'),
                sd_amplitude_norm=('mean_amplitude_norm', 'std'),
                mean_peak_velocity=('mean_peak_velocity', 'mean'),
                sd_peak_velocity=('mean_peak_velocity', 'std'),
            )
            .reset_index()
        )
    else:
        group_df = pd.DataFrame(columns=[
            'stimulus_category', 'n_subjects',
            'mean_saccade_count', 'sd_saccade_count',
            'mean_saccade_duration_ms', 'sd_saccade_duration_ms',
            'mean_total_saccade_duration_ms', 'sd_total_saccade_duration_ms',
            'mean_amplitude_norm', 'sd_amplitude_norm',
            'mean_peak_velocity', 'sd_peak_velocity'
        ])

    # save CSVs
    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_level_saccade_metrics_viewing.csv")
    group_path = os.path.join(save_dir, "group_level_saccade_metrics_viewing.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


def plot_saccadic_metrics_viewing(df, out_dir="saccadic_metrics_viewing_plots"):
    os.makedirs(out_dir, exist_ok=True)

    # categories on x-axis
    categories = ["positive", "negative", "neutral"]
    metrics = [
        ("mean_saccade_count", "Mean saccade count"),
        ("mean_saccade_duration_ms", "Mean saccade duration (ms)"),
        ("mean_total_saccade_duration_ms", "Mean total saccade duration (ms)"),
        ("mean_amplitude_norm", "Mean amplitude (norm)"),
        ("mean_peak_velocity", "Mean peak velocity")
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
        out_path = os.path.join(out_dir, out_fname)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("Saved:", out_path)


def compute_saccadic_metrics_rating(base_dir, save_dir="saccade_metrics_rating"):
    subjects = os.listdir(base_dir)
    subjects = [os.path.join(base_dir, s) for s in subjects]

    subject_summaries = []

    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        df_path = os.path.join(s, "eye_tracking_data.csv")
        if not os.path.exists(df_path):
            continue

        df = pd.read_csv(df_path)
        df.columns = df.columns.str.strip()
        df['stim_cat'] = df['stim_cat'].astype(str).str.strip().replace({'nan': np.nan})

        rating_segs = extract_emotion_rating_segments(df)

        # container: stimulus_category -> emotion_rating -> list of per-stim metrics
        subj_cat_emotion_metrics = defaultdict(lambda: defaultdict(list))

        for stim_id, emotions_dict in rating_segs.items():
            for emotion_rating, seg_df in emotions_dict.items():
                # resolve stimulus category
                stim_cat = seg_df['stim_cat'].iloc[0]
                if pd.isna(stim_cat) or str(stim_cat).strip() == "":
                    fallback = df.loc[df['stim_id'] == stim_id, 'stim_cat']
                    if not fallback.empty and fallback.dropna().any():
                        stim_cat = fallback.ffill().bfill().dropna().astype(str).str.strip().iloc[0]
                    else:
                        stim_cat = "UNKNOWN"
                stim_cat = str(stim_cat).strip()

                # replace invalid stretches with NaN
                seg_df_clean = replace_invalids_w_nans(seg_df, verbose=False)

                # detect saccades from this segment dataframe
                try:
                    sacc_df, G, meta = detect_saccades(
                        seg_df_clean,
                        ts_col='device_time_stamp',
                        ts_unit_microsec=True,
                        fs=1200.0,
                        smoothing_window_samples=7,
                        vel_threshold_k=6.0,
                        min_duration_ms=10.0,
                        max_duration_ms=150.0,
                        min_amplitude_norm=0.02,
                        max_amplitude_norm=1.2,
                        merge_gap_ms=5.0,
                        interp_limit_s=0.02
                    )
                except Exception as e:
                    # skip this segment on error
                    print(f"Error detecting saccades for {subject_id} stim {stim_id}: {e}")
                    continue

                if sacc_df is None or sacc_df.empty:
                    continue

                # per-segment saccade metrics
                sacc_count = len(sacc_df)
                total_duration = float(sacc_df['duration_ms'].sum()) if 'duration_ms' in sacc_df.columns else np.nan
                mean_duration = float(sacc_df['duration_ms'].mean()) if 'duration_ms' in sacc_df.columns else np.nan
                mean_amplitude = float(sacc_df['amplitude_norm'].mean()) if 'amplitude_norm' in sacc_df.columns else np.nan
                mean_peak_velocity = float(sacc_df['peak_velocity_norm_per_s'].mean()) if 'peak_velocity_norm_per_s' in sacc_df.columns else np.nan

                subj_cat_emotion_metrics[stim_cat][emotion_rating].append({
                    'saccade_count': sacc_count,
                    'total_duration': total_duration,
                    'mean_duration': mean_duration,
                    'mean_amplitude': mean_amplitude,
                    'mean_peak_velocity': mean_peak_velocity
                })

        # aggregate per stimulus_category × emotion_rating for this subject
        for stim_cat, emotion_dict in subj_cat_emotion_metrics.items():
            for emotion_rating, metrics_list in emotion_dict.items():
                if not metrics_list:
                    continue

                saccade_counts = np.array([m['saccade_count'] for m in metrics_list], dtype=float)
                total_durations = np.array([m['total_duration'] for m in metrics_list], dtype=float)
                mean_durations = np.array([m['mean_duration'] for m in metrics_list], dtype=float)
                mean_amplitudes = np.array([m['mean_amplitude'] for m in metrics_list], dtype=float)
                mean_peak_vels = np.array([m['mean_peak_velocity'] for m in metrics_list], dtype=float)

                subject_summaries.append({
                    'subject_id': subject_id,
                    'stimulus_category': stim_cat,
                    'emotion_rating': emotion_rating,
                    'mean_saccade_count': float(np.nanmean(saccade_counts)),
                    'sd_saccade_count': float(np.nanstd(saccade_counts, ddof=0)),
                    'mean_saccade_duration_ms': float(np.nanmean(mean_durations)),
                    'sd_saccade_duration_ms': float(np.nanstd(mean_durations, ddof=0)),
                    'mean_total_saccade_duration_ms': float(np.nanmean(total_durations)),
                    'sd_total_saccade_duration_ms': float(np.nanstd(total_durations, ddof=0)),
                    'mean_amplitude_norm': float(np.nanmean(mean_amplitudes)),
                    'sd_amplitude_norm': float(np.nanstd(mean_amplitudes, ddof=0)),
                    'mean_peak_velocity': float(np.nanmean(mean_peak_vels)),
                    'sd_peak_velocity': float(np.nanstd(mean_peak_vels, ddof=0))
                })

    # convert to DataFrame
    subject_df = pd.DataFrame(subject_summaries)

    # group-level summaries aggregated by stimulus_category × emotion_rating
    if not subject_df.empty:
        group_df = (
            subject_df.groupby(['stimulus_category', 'emotion_rating'])
            .agg(
                mean_saccade_count=('mean_saccade_count', 'mean'),
                sd_saccade_count=('mean_saccade_count', 'std'),
                mean_saccade_duration_ms=('mean_saccade_duration_ms', 'mean'),
                sd_saccade_duration_ms=('mean_saccade_duration_ms', 'std'),
                mean_total_saccade_duration_ms=('mean_total_saccade_duration_ms', 'mean'),
                sd_total_saccade_duration_ms=('mean_total_saccade_duration_ms', 'std'),
                mean_amplitude_norm=('mean_amplitude_norm', 'mean'),
                sd_amplitude_norm=('mean_amplitude_norm', 'std'),
                mean_peak_velocity=('mean_peak_velocity', 'mean'),
                sd_peak_velocity=('mean_peak_velocity', 'std'),
            )
            .reset_index()
        )
    else:
        group_df = pd.DataFrame(columns=[
            'stimulus_category', 'emotion_rating',
            'mean_saccade_count', 'sd_saccade_count',
            'mean_saccade_duration_ms', 'sd_saccade_duration_ms',
            'mean_total_saccade_duration_ms', 'sd_total_saccade_duration_ms',
            'mean_amplitude_norm', 'sd_amplitude_norm',
            'mean_peak_velocity', 'sd_peak_velocity'
        ])

    # save CSVs
    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_level_saccade_metrics_per_screen.csv")
    group_path = os.path.join(save_dir, "group_level_saccade_metrics_per_screen.csv")

    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


def plot_saccadic_metrics_rating(df, out_dir="saccadic_metrics_rating_plots"):
    os.makedirs(out_dir, exist_ok=True)

    emotions = ["happy", "sad", "anger", "disgust", "fear"]
    categories = ["positive", "negative", "neutral"]
    metrics = [
        ("mean_saccade_count", "Mean saccade count"),
        ("mean_saccade_duration_ms", "Mean saccade duration (ms)"),
        ("mean_total_saccade_duration_ms", "Mean total saccade duration (ms)"),
        ("mean_amplitude_norm", "Mean amplitude (norm)"),
        ("mean_peak_velocity", "Mean peak velocity")
    ]

    # Visual spacing and sizing controls (tweakable)
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
