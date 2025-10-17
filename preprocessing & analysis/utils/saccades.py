import os
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from .common import extract_emotion_rating_segments, replace_invalids_w_nans

def detect_saccades_from_df(df,
                            ts_col='device_time_stamp',
                            ts_unit_microsec=True,
                            fs=1200.0,
                            smoothing_window_samples=7,
                            vel_threshold_k=6.0,
                            min_duration_ms=10,
                            max_duration_ms=150,
                            min_amplitude_norm=0.02,
                            max_amplitude_norm=1.2,
                            merge_gap_ms=5.0,
                            interp_limit_s=0.02):
    """
    Detect saccades from a DataFrame. Returns (sacc_df, G, meta).
    Assumes normalized screen coords [0..1] for amplitude thresholds.
    - If ts_col present, it will be used (divide by 1e6 if ts_unit_microsec).
    - Otherwise time is constructed from fs and sample index.
    - interp_limit_s controls maximum gap (in seconds) to interpolate across.
    """
    df = df.copy()

    # build gaze trace (uses parse_tuple_cell inside)
    gx, gy = build_gaze_trace(df)

    # timestamps to seconds
    if ts_col in df.columns:
        ts = df[ts_col].astype(float).values
        t = ts / 1e6 if ts_unit_microsec else ts.astype(float)
    else:
        # construct from fs
        t = np.arange(len(df)) / float(fs)

    # ensure monotonic
    if np.any(np.diff(t) <= 0):
        # fallback: rebuild from index and fs
        t = np.arange(len(df)) / float(fs)

    # estimate fs from timestamps (robust)
    dt = np.diff(t)
    if len(dt) == 0:
        raise ValueError("Not enough samples.")
    median_dt = np.median(dt)
    fs_est = 1.0 / median_dt if median_dt > 0 else fs

    # assemble DataFrame
    G = pd.DataFrame({'t': t, 'gx': gx, 'gy': gy})

    # interpolate small NaN gaps but leave long gaps as NaN
    interp_limit = int(max(1, round(interp_limit_s * fs_est)))
    G['gx'] = G['gx'].interpolate(limit=interp_limit, limit_direction='both')
    G['gy'] = G['gy'].interpolate(limit=interp_limit, limit_direction='both')

    # smoothing: rolling median (robust)
    w = max(3, int(smoothing_window_samples) | 1)
    G['gx_s'] = G['gx'].rolling(window=w, center=True, min_periods=1).median()
    G['gy_s'] = G['gy'].rolling(window=w, center=True, min_periods=1).median()

    gx_arr = G['gx_s'].values
    gy_arr = G['gy_s'].values
    t_arr = G['t'].values
    N = len(gx_arr)

    # compute velocity using central difference; produce NaN where neighbors are NaN
    v = np.full(N, np.nan, dtype=float)
    for i in range(1, N-1):
        if np.isnan(gx_arr[i-1]) or np.isnan(gx_arr[i+1]) or np.isnan(gy_arr[i-1]) or np.isnan(gy_arr[i+1]):
            v[i] = np.nan
            continue
        dt_c = t_arr[i+1] - t_arr[i-1]
        if dt_c <= 0:
            v[i] = np.nan
            continue
        vx = (gx_arr[i+1] - gx_arr[i-1]) / dt_c
        vy = (gy_arr[i+1] - gy_arr[i-1]) / dt_c
        v[i] = np.sqrt(vx*vx + vy*vy)
    # ends: use forward/backward diff if possible
    if N >= 2:
        if not (np.isnan(gx_arr[0]) or np.isnan(gx_arr[1]) or np.isnan(gy_arr[0]) or np.isnan(gy_arr[1])):
            dt0 = t_arr[1] - t_arr[0]
            if dt0 > 0:
                v[0] = np.sqrt(((gx_arr[1]-gx_arr[0])/dt0)**2 + ((gy_arr[1]-gy_arr[0])/dt0)**2)
        if not (np.isnan(gx_arr[-1]) or np.isnan(gx_arr[-2]) or np.isnan(gy_arr[-1]) or np.isnan(gy_arr[-2])):
            dtn = t_arr[-1] - t_arr[-2]
            if dtn > 0:
                v[-1] = np.sqrt(((gx_arr[-1]-gx_arr[-2])/dtn)**2 + ((gy_arr[-1]-gy_arr[-2])/dtn)**2)

    G['vel'] = v

    # velocity threshold
    vel_med = np.nanmedian(v)
    vel_mad = mad(v)
    vel_threshold = vel_med + vel_threshold_k * vel_mad
    if vel_mad == 0 or np.isnan(vel_mad):
        vel_threshold = np.nanpercentile(v[~np.isnan(v)], 95) if np.any(~np.isnan(v)) else np.inf

    sacc_mask = (G['vel'] > vel_threshold).astype(int)

    # merge short gaps
    if merge_gap_ms is not None and merge_gap_ms > 0:
        merge_gap_s = merge_gap_ms / 1000.0
        segments = []
        in_s = False
        for i, val in enumerate(sacc_mask):
            if val and not in_s:
                start = i
                in_s = True
            if not val and in_s:
                end = i-1
                segments.append((start, end))
                in_s = False
        if in_s:
            segments.append((start, len(sacc_mask)-1))
        merged = []
        for seg in segments:
            if not merged:
                merged.append(list(seg))
            else:
                prev = merged[-1]
                gap = (t_arr[seg[0]] - t_arr[prev[1]])
                if gap <= merge_gap_s:
                    prev[1] = seg[1]
                else:
                    merged.append(list(seg))
        sacc_mask = np.zeros_like(sacc_mask)
        for a,b in merged:
            sacc_mask[a:b+1] = 1

    # extract saccades
    min_dur_s = max(0.001, min_duration_ms / 1000.0)
    max_dur_s = max_duration_ms / 1000.0
    saccades = []
    i = 0
    while i < N:
        if sacc_mask[i] == 1:
            start = i
            while i < N and sacc_mask[i] == 1:
                i += 1
            end = i-1
            onset_t = t_arr[start]
            offset_t = t_arr[end]
            dur = (offset_t - onset_t) * 1000.0
            if (offset_t - onset_t) < min_dur_s:
                continue
            if (offset_t - onset_t) > max_dur_s:
                continue
            start_pos = np.array([gx_arr[start], gy_arr[start]])
            end_pos = np.array([gx_arr[end], gy_arr[end]])
            if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                continue
            amplitude = float(np.linalg.norm(end_pos - start_pos))
            if amplitude < min_amplitude_norm or amplitude > max_amplitude_norm:
                continue
            peak_v = float(np.nanmax(v[start:end+1]))
            dx, dy = (end_pos - start_pos)
            direction_deg = float(np.degrees(np.arctan2(dy, dx)))
            saccades.append({
                'onset_time_s': float(onset_t),
                'offset_time_s': float(offset_t),
                'duration_ms': float(dur),
                'amplitude_norm': amplitude,
                'peak_velocity_norm_per_s': peak_v,
                'start_x': float(start_pos[0]),
                'start_y': float(start_pos[1]),
                'end_x': float(end_pos[0]),
                'end_y': float(end_pos[1]),
                'direction_deg': direction_deg
            })
        else:
            i += 1

    sacc_df = pd.DataFrame(saccades)
    meta = {'fs_est_hz': fs_est, 'vel_threshold': vel_threshold, 'vel_median': vel_med, 'vel_mad': vel_mad}
    return sacc_df, G, meta


def compute_saccade_metrics_collapsed_0(base_dir, save_dir="saccade_metrics_collapsed_0"):
    """
    Same caller as before but uses detect_saccades_from_df on each segment DataFrame.
    """
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
        category_metrics = defaultdict(list)

        for stim_id, emotions_dict in rating_segs.items():
            for emotion, seg_df in emotions_dict.items():
                # category resolution (same logic as before)
                cat = seg_df['stim_cat'].iloc[0]
                if pd.isna(cat) or str(cat).strip() == "":
                    fallback = df.loc[df['stim_id'] == stim_id, 'stim_cat']
                    if not fallback.empty and fallback.dropna().any():
                        cat = (
                            fallback.ffill().bfill().dropna().astype(str).str.strip().iloc[0]
                        )
                    else:
                        cat = "UNKNOWN"
                else:
                    cat = str(cat).strip()
                emotion_category = cat

                # replace invalid stretches with NaN
                seg_df_clean = replace_invalids_w_nans(seg_df)

                # detect saccades directly from the dataframe
                try:
                    sacc_df, G, meta = detect_saccades_from_df(
                        seg_df_clean,
                        ts_col='device_time_stamp',
                        ts_unit_microsec=True,
                        fs=1200.0,
                        smoothing_window_samples=7,
                        vel_threshold_k=6.0,
                        min_duration_ms=8.0,
                        max_duration_ms=200.0,
                        min_amplitude_norm=0.02,
                        max_amplitude_norm=1.2,
                        merge_gap_ms=5.0,
                        interp_limit_s=0.02
                    )
                except Exception as e:
                    print(f"Error detecting saccades for {subject_id} {stim_id}: {e}")
                    continue

                if not sacc_df.empty:
                    category_metrics[emotion_category].append({
                        "saccade_count": len(sacc_df),
                        "mean_duration_ms": float(sacc_df["duration_ms"].mean()),
                        "mean_amplitude_norm": float(sacc_df["amplitude_norm"].mean()),
                        "mean_peak_velocity": float(sacc_df["peak_velocity_norm_per_s"].mean()),
                    })

        # subject-level aggregation
        for category, metrics_list in category_metrics.items():
            sacc_counts = np.array([m["saccade_count"] for m in metrics_list]) if metrics_list else np.array([0])
            durations = np.array([m["mean_duration_ms"] for m in metrics_list]) if metrics_list else np.array([np.nan])
            amplitudes = np.array([m["mean_amplitude_norm"] for m in metrics_list]) if metrics_list else np.array([np.nan])
            velocities = np.array([m["mean_peak_velocity"] for m in metrics_list]) if metrics_list else np.array([np.nan])

            subject_summaries.append({
                "subject_id": subject_id,
                "emotion_category": category,
                "mean_saccade_count": float(np.nanmean(sacc_counts)),
                "mean_duration_ms": float(np.nanmean(durations)),
                "mean_amplitude_norm": float(np.nanmean(amplitudes)),
                "mean_peak_velocity": float(np.nanmean(velocities)),
            })

    subject_df = pd.DataFrame(subject_summaries)

    group_df = (
        subject_df.groupby("emotion_category")
        .agg(
            mean_saccade_count=("mean_saccade_count", "mean"),
            sd_saccade_count=("mean_saccade_count", "std"),
            mean_duration_ms=("mean_duration_ms", "mean"),
            sd_duration_ms=("mean_duration_ms", "std"),
            mean_amplitude_norm=("mean_amplitude_norm", "mean"),
            sd_amplitude_norm=("mean_amplitude_norm", "std"),
            mean_peak_velocity=("mean_peak_velocity", "mean"),
            sd_peak_velocity=("mean_peak_velocity", "std"),
        )
        .reset_index()
    )

    os.makedirs(save_dir, exist_ok=True)
    subject_path = os.path.join(save_dir, "subject_level_saccade_metrics_collapsed.csv")
    group_path = os.path.join(save_dir, "group_level_saccade_metrics_collapsed.csv")
    subject_df.to_csv(subject_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"Saved subject-level metrics to: {subject_path}")
    print(f"Saved group-level metrics to: {group_path}")

    return subject_df, group_df


def parse_tuple_cell(cell):
    """Parse a stringified tuple like '(0.12, 0.34)' -> (float, float)."""
    if pd.isna(cell) or cell == "":
        return (np.nan, np.nan)
    try:
        t = ast.literal_eval(cell)
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return (float(t[0]), float(t[1]))
    except Exception:
        pass
    return (np.nan, np.nan)

def build_gaze_trace(df,
                     left_xy_col='left_gaze_point_on_display_area',
                     right_xy_col='right_gaze_point_on_display_area',
                     left_valid_col='left_gaze_point_validity',
                     right_valid_col='right_gaze_point_validity'):
    """
    Returns (gx, gy) arrays (float) for gaze in normalized display coords.
    - binocular average where both eyes valid (==1).
    - else use the valid eye.
    - else NaN.
    """
    # parse tuple columns to two-column DataFrames
    left = df[left_xy_col].apply(parse_tuple_cell).apply(pd.Series).rename(columns={0:'lx',1:'ly'})
    right = df[right_xy_col].apply(parse_tuple_cell).apply(pd.Series).rename(columns={0:'rx',1:'ry'})

    # validity columns may be absent; default to zeros if missing
    if left_valid_col in df.columns:
        valid_l = df[left_valid_col].astype(float)
    else:
        valid_l = pd.Series(0.0, index=df.index)

    if right_valid_col in df.columns:
        valid_r = df[right_valid_col].astype(float)
    else:
        valid_r = pd.Series(0.0, index=df.index)

    # combine: binocular average when both valid, else single eye, else NaN
    gx = np.where((valid_l == 1) & (valid_r == 1),
                  (left['lx'].values + right['rx'].values) / 2.0,
                  np.where(valid_l == 1,
                           left['lx'].values,
                           np.where(valid_r == 1,
                                    right['rx'].values,
                                    np.nan)))
    gy = np.where((valid_l == 1) & (valid_r == 1),
                  (left['ly'].values + right['ry'].values) / 2.0,
                  np.where(valid_l == 1,
                           left['ly'].values,
                           np.where(valid_r == 1,
                                    right['ry'].values,
                                    np.nan)))
    return gx.astype(float), gy.astype(float)

def mad(x):
    """
    Median absolute deviation (robust). Returns median(|x - median(x)|).
    Uses nan-aware operations.
    """
    x = np.asarray(x)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

