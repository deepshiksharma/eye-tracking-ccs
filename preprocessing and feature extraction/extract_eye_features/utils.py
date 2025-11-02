import numpy as np


def remove_invalids(dataframe, threshold=1, pad_before=150, pad_after=150, verbose=True):
    """
    Removes continuous sequences of data where the pupil is invalid for atleast `threshold` number of samples;
    as well as the data before and after that sequence (size defined by `padding`).
    
    Args:
        - dataframe (pd.DataFrame):    The dataframe containing eye-tracking data.
        - threshold (int, optional):   Minimum number of continuous invalid rows to trigger removal. Defaults to 1 (aggressive removal).
        - pad_before (int, optional):  Number of rows to remove before each invalid segment. Defaults to 150 (aggressive artifact padding).
        - pad_after (int, optional):   Number of rows to remove after each invalid segment. Defaults to 150 (aggressive artifact padding).
        - verbose (bool, optional):    Print details on how much data has been removed. Defaults to True.
    
    Returns:
        - pd.DataFrame: Dataframe after removing invalid data + padding.
    """
    dataframe = dataframe.copy()

    left_val, right_val = "left_pupil_validity", "right_pupil_validity"
    reset_index = True # reset the dataframe index after dropping rows
    n = len(dataframe)

    if left_val not in dataframe.columns or right_val not in dataframe.columns:
        raise KeyError(f"Columns not found: {left_val}, {right_val}")

    # boolean array: True where either pupil is invalid
    invalid = ((dataframe[left_val] == 0) | (dataframe[right_val] == 0)).to_numpy(dtype=int)

    # find contiguous invalid stretches
    padded = np.concatenate(([0], invalid, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    num_segments = 0
    drop_positions = set()
    for st, ed in zip(starts, ends):
        seg_len = ed - st + 1
        if seg_len >= threshold:
            num_segments += 1  # count this segment
            # drop segment itself
            drop_positions.update(range(st, ed + 1))
            # drop padding before/after
            pre_start = max(0, st - pad_before)
            post_end = min(n - 1, ed + pad_after)
            drop_positions.update(range(pre_start, st))
            drop_positions.update(range(ed + 1, post_end + 1))

    drop_positions_sorted = sorted(drop_positions)
    drop_index_labels = dataframe.index[drop_positions_sorted].tolist()
    dataframe_clean = dataframe.drop(index=drop_index_labels)
    
    if reset_index:
        dataframe_clean = dataframe_clean.reset_index(drop=True)

    if verbose:
        print("input dataframe size:", dataframe.shape[0])
        print("ouput dataframe size:", dataframe_clean.shape[0])
        print("number of segments removed:", num_segments)
        print("total number of rows removed:", len(drop_positions_sorted))
    
    return dataframe_clean


def replace_invalids_w_nans(dataframe, threshold=1, pad_before=150, pad_after=150, verbose=True):
    """
    Replaces continuous sequences of data where the pupil is invalid for at least `threshold` samples
    with NaN values, including the data before and after that sequence (size defined by padding).

    Args:
        - dataframe (pd.DataFrame):    The dataframe containing eye-tracking data.
        - threshold (int, optional):   Minimum number of continuous invalid rows to trigger NaN replacement. Defaults to 1 (aggressive replacement).
        - pad_before (int, optional):  Number of rows to replace with NaN before each invalid segment. Defaults to 150 (aggressive artifact padding).
        - pad_after (int, optional):   Number of rows to replace with NaN after each invalid segment. Defaults to 150 (aggressive artifact padding).
        - verbose (bool, optional):    Print details on how much data has been repalced. Defaults to True.
    
    Returns:
        - pd.DataFrame: Dataframe with invalid data and padding replaced by NaN.
    """
    dataframe = dataframe.copy()
    
    left_val, right_val = "left_pupil_validity", "right_pupil_validity"
    n = len(dataframe)

    if left_val not in dataframe.columns or right_val not in dataframe.columns:
        raise KeyError(f"Columns not found: {left_val}, {right_val}")

    # boolean array: True where either pupil is invalid
    invalid = ((dataframe[left_val] == 0) | (dataframe[right_val] == 0)).to_numpy(dtype=int)

    # find contiguous invalid stretches
    padded = np.concatenate(([0], invalid, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    num_segments = 0
    replace_positions = set()
    for st, ed in zip(starts, ends):
        seg_len = ed - st + 1
        if seg_len >= threshold:
            num_segments += 1  # count this segment
            # mark segment itself
            replace_positions.update(range(st, ed + 1))
            # mark padding before/after
            pre_start = max(0, st - pad_before)
            post_end = min(n - 1, ed + pad_after)
            replace_positions.update(range(pre_start, st))
            replace_positions.update(range(ed + 1, post_end + 1))

    replace_positions_sorted = sorted(replace_positions)

    # Replace all target rows with NaN across all columns
    dataframe.iloc[replace_positions_sorted] = np.nan

    if verbose:
        print("dataframe size:", dataframe.shape[0])
        print("number of segments replaced with np.nan:", num_segments)
        print("total number of rows replaced with np.nan:", len(replace_positions_sorted))
    
    return dataframe
