import pandas as pd
import pygame
import tobii_research as tr


# Initialize eye tracker
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]

gaze_data = []
def gaze_data_callback(data):
    gaze_data.append(data)


# pygame init
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=0)
screen_width, screen_height = pygame.display.get_window_size()
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# start fix
fix_image = pygame.image.load("./fix.png").convert()
screen.fill(pygame.Color('black'))
image_rect = fix_image.get_rect(center=(screen_width // 2, screen_height // 2))
screen.blit(fix_image, image_rect)
pygame.display.flip()
pygame.time.wait(6_000)


# start eye tracker 
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

# show img
image = pygame.image.load("./Ilya_Repin_Unexpected_visitors.png").convert()
screen.fill(pygame.Color('black'))
image_rect = image.get_rect(center=(screen_width // 2, screen_height // 2))
screen.blit(image, image_rect)
pygame.display.flip()
pygame.time.wait(60_000)

# stop eyetracker
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


# end fix
screen.fill(pygame.Color('black'))
image_rect = fix_image.get_rect(center=(screen_width // 2, screen_height // 2))
screen.blit(fix_image, image_rect)
pygame.display.flip()
pygame.time.wait(6_000)

pygame.quit()

# Save eye tracking data
df = pd.DataFrame.from_dict(gaze_data)
df['seconds'] = df['device_time_stamp'].apply(lambda x: x / 1e6)
df.to_csv("./captured_data/eye_tracking_data.csv", index=False)

print("done.")
