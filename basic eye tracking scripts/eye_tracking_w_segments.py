import pygame
import pandas as pd
import tobii_research as tr


# Initialize eye tracker
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]


# Initialize variables to mark events during the recording
stimulus_present = False
remarks = None


# Initialize master list to hold all eye-tracking data
gaze_data = []


# Callback function to record eye-tracking data
def gaze_data_callback(data):
    # Add the additional event marking variables to eye-tracking data record
    # This ensures precise timing while automatically adding event markers
    data.update(
        {
            'stimulus_present': stimulus_present,
            'remarks': remarks
        }
    )
    gaze_data.append(data)


# Start eye tracker 
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)


# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=0)
screen_width, screen_height = pygame.display.get_window_size()
pygame.mouse.set_visible(False)


# Display fixation cross at start for 5 seconds
fixation = pygame.image.load("./fixation_cross.png").convert()
screen.fill(pygame.Color('black'))
image_rect = fixation.get_rect(center=(screen_width // 2, screen_height // 2))
screen.blit(fixation, image_rect)
pygame.display.flip()
remarks = "FIXATION CROSS" # Update event marking variable
pygame.time.wait(5_000)
remarks = None # Update event marking variable


# Display image on screen for 30 seconds
image = pygame.image.load("./ilya_repin_unexpected_visitors.png").convert()
screen.fill(pygame.Color('black'))
screen.blit(image, image_rect)
pygame.display.flip()
stimulus_present = True # Update event marking variable
pygame.time.wait(30_000)
stimulus_present = False # Update event marking variable


# Display fixation cross at end for 5 seconds
screen.fill(pygame.Color('black'))
screen.blit(fixation, image_rect)
pygame.display.flip()
remarks = "FIXATION CROSS" # Update event marking variable
pygame.time.wait(5_000)
remarks = None # Update event marking variable


# Stop eye tracker
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


# Stop PyGame
pygame.quit()


# Save eye tracking data
df = pd.DataFrame.from_dict(gaze_data)
df.to_csv("eye_tracking_data_w_segments.csv", index=False)

print("Eye-tracking data saved.")
