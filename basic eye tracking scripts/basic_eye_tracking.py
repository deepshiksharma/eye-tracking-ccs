import pygame
import pandas as pd
import tobii_research as tr


"""
START EYE-TRACKING

DISPLAY AN IMAGE ON SCREEN FOR 30 SECONDS

STOP EYE-TRACKING

SAVE DATA
"""


# Initialize eye tracker
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]

# Print eye tracker details
print("Address:", my_eyetracker.address)
print("Model:", my_eyetracker.model)
print("Name (OK if empty):", my_eyetracker.device_name)
print("Serial number:", my_eyetracker.serial_number)


# Initialize master list to hold all eye-tracking data
gaze_data = []

# Callback function to record eye-tracking data
def gaze_data_callback(data):
    gaze_data.append(data)


# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=0)
screen_width, screen_height = pygame.display.get_window_size()
pygame.mouse.set_visible(False)


# Start eye tracker 
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)


# Display image on screen for 30 seconds
image = pygame.image.load("./ilya_repin_unexpected_visitors.png").convert()
screen.fill(pygame.Color('grey'))
image_rect = image.get_rect(center=(screen_width // 2, screen_height // 2))
screen.blit(image, image_rect)
pygame.display.flip()
pygame.time.wait(30_000)


# Stop eye tracker
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


# Stop PyGame
pygame.quit()


# Save eye tracking data
df = pd.DataFrame.from_dict(gaze_data)
df.to_csv("eye_tracking_data_basic.csv", index=False)

print("Eye-tracking data saved.")
