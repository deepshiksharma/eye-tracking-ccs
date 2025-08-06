import os, csv
import pygame
import pandas as pd

def initialize_pygame_screen():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=1)
    screen_width, screen_height = pygame.display.get_window_size()
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    return screen, screen_width, screen_height, clock

def show_image_centered(screen, image, screen_width, screen_height):
    screen.fill(pygame.Color('black'))
    image_rect = image.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(image, image_rect)

def get_images(stimuli_config, stim_scram):
    img_bank = {}
    for subdir in stimuli_config['img_subdirs']:
        path = os.path.join(stimuli_config[stim_scram], subdir)
        img_bank[subdir] = os.listdir(path)
    return img_bank

def save_ratings(csv_path, ratings_row, emotions):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name'] + emotions)
        writer.writerow(ratings_row)

def save_gaze_data(gaze_data, save_dir):
    df = pd.DataFrame.from_dict(gaze_data)
    if 'device_time_stamp' in df.columns:
        df['seconds'] = df['device_time_stamp'].apply(lambda x: x / 1e6)
    output_path = os.path.join(save_dir, "eye_tracking_data.csv")
    df.to_csv(output_path, index=False)
