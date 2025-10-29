import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from compute_eye_metrics.utils import remove_invalids
from compute_eye_metrics.pupildiameter import apply_smoothing, baseline_correction


def compute_mean_sd(base_dir, save_csv=True, plot=True):   
    subjects = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    subject_summaries = []
    for s in subjects:
        subject_id = os.path.splitext(os.path.basename(s))[0]
        
        df = pd.read_csv(os.path.join(s, "eye_tracking_data.csv"))

        # clean, smooth, and baseline correct
        df = remove_invalids(df, verbose=False)
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
        plt.savefig('Absolute pupil diameter per emotion category.png')
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
        plt.savefig('Baseline-corrected pupil diameter per emotion category.png')
        plt.show()

    return group_lvl, subject_lvl


def run_ttests(subject_lvl, save_txt=True):
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
