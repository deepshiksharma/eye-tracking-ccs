import ast
import numpy as np
import pandas as pd


def parse_tuple_cell(cell):
    """
    Parse a stringified tuple into floats. '(0.12, 0.34)' -> (float, float).
    Helper function for build_gaze_trace.
    """
    if pd.isna(cell) or cell == "":
        return (np.nan, np.nan)
    try:
        t = ast.literal_eval(cell)
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return (float(t[0]), float(t[1]))
    except Exception:
        pass
    return (np.nan, np.nan)

def build_gaze_trace(dataframe,
                     left_xy_col='left_gaze_point_on_display_area', right_xy_col='right_gaze_point_on_display_area',
                     left_valid_col='left_gaze_point_validity', right_valid_col='right_gaze_point_validity'):
    """
    Construct a continuous gaze position trace (gx, gy) in normalized screen coordinates.
    Helper function for detect_saccades.
    
        Binocular average where both eyes are valid (==1).
        Otherwise use the valid eye.
        Else NaN.
    """
    df = dataframe.copy()

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
    Helper function for detect_saccades.
    """
    x = np.asarray(x)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def detect_saccades(dataframe,
                    ts_col='device_time_stamp', ts_unit_microsec=True, fs=1200.0,
                    smoothing_window_samples=7, vel_threshold_k=6.0,
                    min_duration_ms=10, max_duration_ms=150, min_amplitude_norm=0.02, max_amplitude_norm=1.2,
                    merge_gap_ms=5.0, interp_limit_s=0.02):
    """
    Extracts saccadic metrics from a given dataframe.
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data.
        - ts_col (str, optional):   Name of the column that contains timestamp. Defaults to 'device_time_stamp'.
        - ts_unit_microsec (bool, optional):    Whether the timestamps are in microseconds. Defaults to True.
        - fs (numeric, optional):   Sampling rate. Defaults to 1200.
        - smoothing_window_samples, vel_threshold_k, min_duration_ms, max_duration_ms,
          min_amplitude_norm, max_amplitude_norm, merge_gap_ms, interp_limit_s: Various optional parameters for saccadic detection.

    Returns:
        - saccades (pd.DataFrame):  The dataframe containing all detected saccades, along with their corresponding metrics.
        - G (pd.DataFrame): The gaze trace used by the saccade detector function.
        - metadata (dict):  Metadata containing a few diagnostic and configuration values used for saccade detection.
    """
    df = dataframe.copy()

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

    saccades = pd.DataFrame(saccades)
    metadata = {'fs_est_hz': fs_est, 'vel_threshold': vel_threshold, 'vel_median': vel_med, 'vel_mad': vel_mad}
    return saccades, G, metadata
