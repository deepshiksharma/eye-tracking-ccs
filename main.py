import os, sys, random, csv
import tobii_research as tr
import pygame
import yaml
from rating_utils import display_rating_screen
from paradigm_utils import (
    initialize_pygame_screen, show_image_centered, get_images, save_ratings, save_gaze_data
)

# Check command line args
if len(sys.argv) != 2:
    print("Incorrect usage.")
    sys.exit("Usage: python main.py <subject_directory_name>")

os.makedirs("SUBJECTS", exist_ok=True)

subj_save_dir = sys.argv[1]
SUBJ_SAVE_DIR = os.path.join("SUBJECTS", subj_save_dir)

try:
    os.mkdir(SUBJ_SAVE_DIR)
except FileExistsError:
    sys.exit(f"Error: Subject directory \"{SUBJ_SAVE_DIR}\" already exists.")
except Exception as e:
    sys.exit(f"Error while creating subject directory: {e}")

with open("config.yaml") as f:
    config = yaml.safe_load(f)


# Initialize eye tracker
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]

gaze_data = []
stim_present, stim_id, remarks = False, None, None

def gaze_data_callback(data):
    data.update({
        'stim_present': stim_present,
        'stim_id': stim_id,
        'remarks': remarks
    })
    gaze_data.append(data)

my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)


def fixation_cross():
    global remarks
    show_image_centered(screen, fixation_img, SCREEN_WIDTH, SCREEN_HEIGHT)
    remarks = 'FIXATION CROSS'
    pygame.display.update()
    pygame.time.wait(config['timing']['fixation_duration'])
    remarks = None


# Setup Pygame
screen, SCREEN_WIDTH, SCREEN_HEIGHT, clock = initialize_pygame_screen(config['display'])
fixation_img = pygame.image.load(config['stimuli']['fixation_path']).convert()

# Load images
stim_img_bank = get_images(config['stimuli'], 'normalized_img_dir')
scram_img_bank = get_images(config['stimuli'], 'scrambled_img_dir')


# START PARADIGM
ratings_path = os.path.join(SUBJ_SAVE_DIR, "ratings.csv")
with open(ratings_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name'] + config['ratings']['emotions'])

fixation_cross()

for _ in range(config['stimuli']['n_trials']):
    
    random.shuffle(config['stimuli']['img_subdirs'])
    for category in config['stimuli']['img_subdirs']:
        
        # Select a random image
        selected_img = random.choice(stim_img_bank[category])
        
        # Show scrambled image
        img_path = os.path.join(config['stimuli']['scrambled_img_dir'], category, selected_img)
        scram_img = pygame.image.load(img_path).convert()

        stim_id, remarks = selected_img, 'SCRAMBLED IMAGE'
        show_image_centered(screen, scram_img, SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.update()
        pygame.time.wait(config['timing']['stimulus_duration'])
        remarks = None

        # Show stimulus image
        img_path = os.path.join(config['stimuli']['normalized_img_dir'], category, selected_img)
        stim_img = pygame.image.load(img_path).convert()

        stim_present = True
        show_image_centered(screen, stim_img, SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.update()
        pygame.time.wait(config['timing']['stimulus_duration'])
        stim_present = False

        # Emotion rating screens
        ratings_row = {'image_name': selected_img}
        emotions = config['ratings']['emotions'].copy()
        
        random.shuffle(emotions)
        for emo in emotions:
            remarks = f'{emo}_EMOTION_RATING'
            rating = display_rating_screen(emo, screen, config, SCREEN_WIDTH, SCREEN_HEIGHT, clock)
            ratings_row[emo] = rating

        save_ratings(ratings_path, ratings_row, config['ratings']['emotions'])
        
        stim_img_bank[category].remove(selected_img)
        scram_img_bank[category].remove(selected_img)
        stim_id, remarks = None, None


fixation_cross()

pygame.quit()

# Save eye tracking data
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
save_gaze_data(gaze_data, SUBJ_SAVE_DIR)
