# Eye Tracking Experiments

**Equipment used**
- Eye tracker: Tobii Pro Spectrum @ 1200 Hz
- Monitor: Eizo Flexscan EV2451, 1920x1080 @ 60 Hz

**Regarding Tobii timestamps**
- Timestamps are in microseconds [[1]].
- Both timestamps (`device_time_stamp` and `system_time_stamp`) are monotonic [[2]].

[1]: https://developer.tobiipro.com/commonconcepts/timestamp-and-timing.html
[2]: https://connect.tobii.com/s/article/What-is-the-difference-between-Device-Timestamp-and-System-Timestamp?language=en_US

device ts? system ts?
...


## Basic eye tracking script
`basic_eye_tracking.py` and `eye_tracking_w_segments.py` demonstrates the barebones required for data collection. It showcases a high-level overview of the various script components which can be modified for a variety of experiment paradigms.


## Visual stimuli paradigm for data collection
Directory: `visual stimuli paradigm/`

This experiment paradigm displays images to the subject, and then asks for their subjective rating of emotions after each image.
...


## Preprocessing & Analysis
Directory: `preprocessing and analysis/`

Functions to extract eye tracking metrics are inside  `compute_eye_metrics/`. The contents of this directory can be imported as a package in Python. 

The following functionality is provided by this package:
- Preprocess pupil diameter signal
    - Remove invalid data (blinks and blink/movement artifacts)
    - Apply smoothing
    - Baseline-correction
- Detect fixations
    - The count and duration of fixations can be computed from detected fixations.
- Detect saccades
    - The count and duration of saccades, as well as the saccadic amplitude and peak saccadic velocity can be computed from detected saccades.

An overview of these functions and their usage is provided in Jupyter notebooks:
- `pupil_diameter.ipynb`
- `fixations.ipynb`
- `saccades.ipynb`

A sample csv file containing raw data from the eye-tracker is also included: `sample_eye_tracking_data.csv`.

---

Experiment specific functions are inside `experiment_specific_utils/`. The scripts inside this directory are used for analysis and plotting (which is specific to `the visual stimuli paradigm/`). The Jupyter notebooks `[experiment_specific]*.ipynb` are also experiment specific; they simply run the functions provided by `experiment_specific_utils/`.

---

Deepshik Sharma <br>
Research Trainee *(Jul - Oct 2025)* <br>
Center for Consciousness Studies, <br>
Department of Neurophysiology, <br>
NIMHANS, Bangalore, India
