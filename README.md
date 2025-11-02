# Eye Tracking Experiments

**Equipment used**
- Eye tracker: Tobii Pro Spectrum @ 1200 Hz
- Monitor: Eizo Flexscan EV2451, 1920x1080 @ 60 Hz

**Regarding Tobii timestamps**
- Timestamps are in microseconds [[1]].
- Both timestamps (`device_time_stamp` and `system_time_stamp`) are monotonic. <br>
`device_time_stamp` is from the internal clock of the eye tracker; `system_time_stamp` uses the CPU clock of the computer on which the application is running [[2]].

[1]: https://developer.tobiipro.com/commonconcepts/timestamp-and-timing.html
[2]: https://connect.tobii.com/s/article/What-is-the-difference-between-Device-Timestamp-and-System-Timestamp?language=en_US


## Basic eye tracking scripts
### Directory: `basic eye tracking scripts/`

`basic_eye_tracking.py` demonstrates the barebones required for an eye-tracking recording.

The approach I used to precisely mark events during the recording is demonstrated in `eye_tracking_w_segments.py`:
```py
gaze_data = []
stimulus_present = False
remarks = None

def gaze_data_callback(data):
    data.update(
        {
            'stimulus_present': stimulus_present,
            'remarks': remarks
        }
    )
    gaze_data.append(data)
```
By adding new key:value pairs to the dict object used by the Tobii data recording function, the script is able to precisely mark events. This can be modified to suit a variety of experiment paradigms.

### Generating gaze heatmaps
Run `generate_gaze_heatmap.py <path_to_eye_tracker_data.csv> <path_to_overlay_image>` to generate a gaze heatmap from a csv file. <br>
A sample heatmap is shown below:

![heatmap](./basic%20eye%20tracking%20scripts/heatmap.png)


## Emotion rating paradigm for data collection
### Directory: `emotion rating paradigm/`

This experiment paradigm displays images to the subject, and then asks for their subjective rating of emotions after each image.

![experiment design](./emotion%20rating%20paradigm/assets/other%20stuff/experiment%20design.png)

- A fixation cross is displayed at the start and end of the experiment session.
- The corresponding scrambled image* is displayed before the actual stimuli image.
- The stimuli image is displayed.
- Following the stimuli, five emotion rating screens are displayed (Happy, Sad, Anger, Disgust, and Fear), each on a scale of 1 to 7. _The subject should select a rating to move on from this section._

_*Scrambled image corresponding to the particular stimuli image that will follow._ <br>
_Durations for each component should be set in `config.yaml`._

### To run this experiment:
- Place your images inside a directory within `assets/`.
_Ensure that each category of stimuli is within its own subdirectory inside this image directory._
    - Images used in this experiment can be found inside `assets/NAPS_nencki.zip`. _This zip file showcases an example of the required directory structure._
- Run `assets/norm_scram_generate-fix.py` to generate contrast balanced stimuli images, their corresponding fourier-phase scrambled images (to be used as baseline image), and a fixation cross.
- Set experiment configuration settings in `config.yaml`. 
- `main.py` loads this configuration file, and runs the experiment as per this configuration.
_`main.py` depends on functions within `paradigm_utils.py` and `rating_utils.py`._

The captured eye-tracking data is saved to `SUBJECTS/<subject_name>/eye_tracking_data.csv`.

This experiment script adds new columns to the csv file: `stim_present`, `stim_cat`, `stim_id`, and `remarks`. <br>
These new columns help in segmenting the data later on during feature extraction and analysis.

As demonstrated by `basic eye tracking scripts/eye_tracking_w_segments.py`, new columns can be added to the eye-tracking data that will keep track of various events that occur during the recording session.


## Preprocessing & Feature Extraction
### Directory: `preprocessing and feature extraction/`

Functions to extract eye tracking features are inside `extract_eye_features/`. The contents of this directory can be imported as a package in Python. 

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

Experiment specific functions are inside `experiment_specific_utils/`. The scripts inside this directory are used for analysis and plotting (which is specific to `emotion rating paradigm/`). <br>

The Jupyter notebooks `[experiment_specific]*.ipynb` are also experiment specific; they simply run the functions provided by `experiment_specific_utils/`.

---

Deepshik Sharma <br>
Research Trainee *(Jul - Nov 2025)* <br>
Center for Consciousness Studies <br>
Department of Neurophysiology, <br>
NIMHANS, Bangalore, India

<img src="./emotion%20rating%20paradigm/assets/other%20stuff/my_eyes.png" alt="my_eyes" width="225"/> <br>
_my eyes, as seen by the eye tracker : )_
