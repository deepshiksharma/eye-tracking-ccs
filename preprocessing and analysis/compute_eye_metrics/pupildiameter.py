import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def apply_smoothing(dataframe, window_length=101, polyorder=3, plot_fig=False):
    """
    Applies smoothing to the pupil diameter to decrease random fluctuations.
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data,
                                    ideally after invalid rows have been removed by the `remove_invalids` function.
        - window_length (int, optional): The size of moving window over which the polynomial is fitted. Argument is passed to Savitzky-Golay filter. Defaults to 101.
        - polyorder (int, optional):     The degree of polynomial used to approximate the data inside each window. Argument is passed to Savitzky-Golay filter. Defaults to 3.
        - plot_fig (bool, optional):     If True, plot figure which overlays the raw and smoothed signals. Defaults to False.
    
    Returns:
        - pd.DataFrame: Dataframe after applying smoothing to pupil diameter.
                                    This dataframe will contain two new columns: "left_pupil_diameter_smooth" and "right_pupil_diameter_smooth".                     
    """
    dataframe = dataframe.copy()

    # This function applies the Savitzky-Golay filter to smooth a column containing the pupil diameter.
    def apply_smoothing_per_pupil(pupil_diameter_column):
        x = np.arange(len(pupil_diameter_column))
        y = signal.savgol_filter(pupil_diameter_column, window_length=window_length, polyorder=polyorder)
        return x, y

    left = dataframe.left_pupil_diameter
    right = dataframe.right_pupil_diameter

    leftx, lefty = apply_smoothing_per_pupil(left)
    rightx, righty = apply_smoothing_per_pupil(right)

    if plot_fig == True:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(left, color='b', alpha=0.5, label='Raw signal')
        plt.plot(leftx, lefty, color='r', label='Smoothed signal')
        plt.title("Left pupil diameter")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(right, color='b', alpha=0.5, label='Raw signal')
        plt.plot(rightx, righty, color='r', label='Smoothed signal')
        plt.title("Right pupil diameter")
        plt.legend()
        plt.tight_layout()
        plt.show()

    dataframe["left_pupil_diameter_smooth"] = lefty
    dataframe["right_pupil_diameter_smooth"] = righty
    
    return dataframe


def baseline_correction(dataframe):
    """
    Performs subtractive baseline correction (per pupil, per trial) by taking the mean from a 500 ms baseline period (scrambled image).
    
    Args:
        - dataframe (pd.DataFrame): The dataframe containing eye-tracking data,
                             ideally after removing invalid rows using `remove_invalids`,
                             and smoothing the pupil diameter signal using `apply_smoothing`.

    Returns:
        - pd.DataFrame: Dataframe after performing baseline correction.
                                    This dataframe will contain two new columns: "left_pupil_diameter_bc" and "right_pupil_diameter_bc".
    """
    dataframe = dataframe.copy()

    # columns for baseline-corrected values
    dataframe["left_pupil_diameter_bc"] = np.nan
    dataframe["right_pupil_diameter_bc"] = np.nan

    # group by stim_id (one trial per image)
    for stim_id, trial_df in dataframe.groupby("stim_id"):
        if pd.isna(stim_id):
            continue
        
        # scrambled image rows before the stimulus
        scram_rows = trial_df[trial_df["remarks"] == "SCRAMBLED IMAGE"]
        if scram_rows.empty:
            continue
        
        # 500 ms baseline
        baseline_rows = scram_rows.tail(1000).head(600)

        left_base = baseline_rows["left_pupil_diameter_smooth"].mean()
        right_base = baseline_rows["right_pupil_diameter_smooth"].mean()
        
        # all rows after scrambled image until the next fixation cross
        mask = (dataframe["stim_id"] == stim_id) & (dataframe["remarks"] != "SCRAMBLED IMAGE")
        
        dataframe.loc[mask, "left_pupil_diameter_bc"] = (
            dataframe.loc[mask, "left_pupil_diameter_smooth"] - left_base
        )
        dataframe.loc[mask, "right_pupil_diameter_bc"] = (
            dataframe.loc[mask, "right_pupil_diameter_smooth"] - right_base
        )

    return dataframe
