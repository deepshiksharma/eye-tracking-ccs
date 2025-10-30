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


## Basic eye tracking scripts
Directory: `basic eye tracking scripts/`

`basic_eye_tracking.py` and `eye_tracking_w_segments.py` demonstrates the barebones required for data collection. <br>
It showcases a high-level overview of the various script components which can be modified for a variety of experiment paradigms.

To generate a gaze heatmap from a csv file, run `generate_gaze_heatmap.py`. A sample gaze heatmap is shown below:

[ADD GAZE HEATMAP HERE]


## Visual stimuli paradigm for data collection
Directory: `visual stimuli paradigm/`

This experiment paradigm displays images to the subject, and then asks for their subjective rating of emotions after each image.

![Experiment design](./visual%20stimuli%20paradigm/assets/other%20stuff/experiment%20design.png)

- A fixation cross is displayed at the start and end of the experiment session.
- The corresponding* scrambled image is displayed before the actual stimuli image.
- The stumuli image is displayed.
- Following the stimuli, five emotion rating screens are displayed (Happy, Sad, Anger, Disgust, and Fear), each on a scale of 1 to 7. _The subject should select a rating to move on the the next section._

_*Corresponding to the particular stimuli image that will follow._ <br>
_Durations for each should be set in `config.yaml`._

To run this experiment:
- Place your images inside a dir within `assets/`.
_Ensure that each category of stimuli is within its own subdir inside this image dir._
- Images used in this experiment can be found inside `assets/NAPS_nencki.zip`. _Extract this and refer to its directory structure as an example._
- Run `assets/norm_scram_generate-fix.py` to generate contrast balanced stimuli images, and their corresponding fourier-phase scrambled images (to be used as baseline image).
- Set experiment configuration settings in `config.yaml`. 
- `main.py` loads this configuration file, and runs the experiment as per this configuation.
_`main.py` depends on functions within `paradigm_utils.py` and `rating_utils.py`._

The captured eye-tracker data is saved to `SUBJECTS/<subject_name>/eye_tracking_data.csv`.

This experiment script adds new columns to the csv file: `stim_present`, `stim_cat`, `stim_id`, and `remarks`. <br>
These new columns help in segmenting the data later on during processing and analysis. <br>
As demonstrated in `basic eye tracking scripts/eye_tracking_w_segments.py`, new columns can be added to the eye-tracking data that will keep track of various events that occur during the experiment session.


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

An overview of these functions and their usage is provided in the following Jupyter notebooks:
- `pupil_diameter.ipynb`
- `fixations.ipynb`
- `saccades.ipynb`

A sample csv file containing raw data from the eye-tracker is also included: `sample_eye_tracking_data.csv`.

---

Experiment specific functions are inside `experiment_specific_utils/`. The scripts inside this directory are used for analysis and plotting (which is specific to `the visual stimuli paradigm/`). <br>
The Jupyter notebooks `[experiment_specific]*.ipynb` are also experiment specific; they simply run the functions provided by `experiment_specific_utils/`.

---

Deepshik Sharma <br>
Research Trainee *(Jul - Oct 2025)* <br>
Center for Consciousness Studies, <br>
Department of Neurophysiology, <br>
NIMHANS, Bangalore, India
