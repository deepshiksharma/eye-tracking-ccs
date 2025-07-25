import os, random, time, sys, csv
import tobii_research as tr
import pandas as pd
import pygame


# STIMULI CONFIG
IMG_MASTER_DIR = r".\NAPS_nencki"
IMG_SUBDIRS = ['Neutral', 'PositiveLow', 'NegativeHigh']
FIXATION_PATH = r".\fixation.png"
CHEESE_PATH = r".\cheese.png"

# EMOTION RATING SCALE CONFIG
ICON_DIR = r".\icons"
EMOTIONS = ['happy', 'sad', 'anger', 'fear', 'disgust']
RATING_LABELS_TEMPLATE = ['No emotion', 'Neutral', None, 'Moderately', 'Maximum']

emotion_rating_image_order = {
    'happy': ['no_emotion', 'neutral', 'happy_5', 'happy_8', 'happy_10'],
    'sad': ['no_emotion', 'neutral', 'sad_5', 'sad_8', 'sad_10'],
    'anger': ['no_emotion', 'neutral', 'anger_5', 'anger_8', 'anger_10'],
    'fear': ['no_emotion', 'neutral', 'fear_5', 'fear_8', 'fear_10'],
    'disgust': ['no_emotion', 'neutral', 'disgust_5', 'disgust_8', 'disgust_10']
}


# EYE TRACKER SETUP
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]

gaze_data = []

stimulus_present, stimulus_id = False, None
remarks = None

def gaze_data_callback(data):
    global stimulus_present, stimulus_id, remarks, gaze_data
    data['stimulus_present'] = stimulus_present
    data['stimulus_id'] = stimulus_id
    data['remarks'] = remarks
    gaze_data.append(data)


# START EYE TRACKER
my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
time.sleep(1)


# PYGAME
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=1)
SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_window_size()
BG_COLOR = pygame.Color('black')
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 64)
label_font = pygame.font.SysFont(None, 36)

BASE_RADIUS = 25
HOVER_RADIUS = 30
LINE_Y = SCREEN_HEIGHT // 2
NUM_POINTS = 9
GAP = SCREEN_WIDTH // (NUM_POINTS + 1)
FPS = 60

# Load fixation cross and cheese image 
fixation_cross = pygame.image.load(FIXATION_PATH).convert()
cheese = pygame.image.load(CHEESE_PATH).convert()


# EMOTION RATING UTIL FUNCTIONS
def get_points():
    points = []
    for i in range(NUM_POINTS):
        x = GAP * (i + 1)
        y = LINE_Y
        points.append(pygame.Rect(x - HOVER_RADIUS, y - HOVER_RADIUS, HOVER_RADIUS * 2, HOVER_RADIUS * 2))
    return points

def load_emoji_images(names, size=(125, 125)):
    images = []
    for name in names:
        path = os.path.join(ICON_DIR, f"{name}.png")
        try:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, size)
            images.append(img)
        except:
            images.append(None)
    return images

def display_rating_screen(emotion, output_csv_path):
    selected_rating = None
    points = get_points()
    running = True
    RATING_LABELS = [emotion.capitalize() if lbl is None else lbl for lbl in RATING_LABELS_TEMPLATE]
    emoji_names = emotion_rating_image_order[emotion]
    emoji_images = load_emoji_images(emoji_names)

    while running:
        screen.fill(BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True

        # Title
        title_text = font.render(f"Rate the emotion: {emotion.upper()}", True, (255, 255, 255))
        screen.blit(title_text, ((SCREEN_WIDTH - title_text.get_width()) // 2, 100))

        # Draw line and points
        pygame.draw.line(screen, (255, 255, 255), (GAP, LINE_Y), (SCREEN_WIDTH - GAP, LINE_Y), 2)

        for i, rect in enumerate(points):
            is_hovered = rect.collidepoint(mouse_pos)
            point_x = rect.centerx
            point_y = rect.centery
            radius = HOVER_RADIUS if is_hovered else BASE_RADIUS
            color = (255, 255, 255)
            pygame.draw.circle(screen, color, (point_x, point_y), radius)
            number_label = label_font.render(str(i + 1), True, color)
            screen.blit(number_label, (point_x - number_label.get_width() // 2, LINE_Y + 40))
            if is_hovered and click and selected_rating is None:
                selected_rating = i + 1
                with open(output_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([emotion, selected_rating])
                running = False

        emoji_positions = [0, 2, 4, 6, 8]
        for idx, point_index in enumerate(emoji_positions):
            if idx < len(emoji_images) and emoji_images[idx]:
                img = emoji_images[idx]
                x = points[point_index].centerx
                y = LINE_Y - img.get_height() - 80
                screen.blit(img, (x - img.get_width() // 2, y))

        for idx, point_index in enumerate(emoji_positions):
            label_text = RATING_LABELS[idx]
            label = label_font.render(label_text, True, (200, 200, 200))
            x = points[point_index].centerx
            y = LINE_Y + 90
            screen.blit(label, (x - label.get_width() // 2, y))

        pygame.mouse.set_visible(True)
        pygame.display.flip()
        clock.tick(FPS)

        pygame.mouse.set_visible(False)


# PARADIGM START
"""
FIXATION CROSS (START) -> 6s
    
    for N in Valence_Image_Categories:
        CHEESE IMAGE -> 3s
        STIMULUS IMAGE -> 5s
        EMOTION RATING SCREEN (rate intensity of all 5 emotions, per image)

FIXATION CROSS (END) -> 6s
"""

# FIXATION CROSS (START) -> 6s
image_frame = fixation_cross.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
screen.blit(fixation_cross, image_frame)

remarks = 'FIXATION CROSS'
pygame.display.update()
pygame.time.wait(6000)
remarks = None

random.shuffle(IMG_SUBDIRS)
for emotion_dir in IMG_SUBDIRS:
    
    # CHEESE IMAGE -> 3s
    screen.fill(BG_COLOR)
    image_frame = cheese.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
    screen.blit(cheese, image_frame)
    
    remarks = 'CHEESE IMAGE'
    pygame.display.update()
    pygame.time.wait(3000)
    remarks = None

    # SELECT STIMULUS IMAGE
    images = os.listdir(os.path.join(IMG_MASTER_DIR, emotion_dir))
    random.shuffle(images)
    selected_img = random.choice(images)
    image_path = os.path.join(IMG_MASTER_DIR, emotion_dir, selected_img)

    # SHOW STIMULUS
    stimulus_image = pygame.image.load(image_path).convert()
    screen.fill(BG_COLOR)
    image_frame = stimulus_image.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
    screen.blit(stimulus_image, image_frame)

    stimulus_present, stimulus_id = True, selected_img
    pygame.display.update()
    pygame.time.wait(5000)
    stimulus_present = False

    # Prompt ratings for all emotions (store per-image CSV)
    random.shuffle(EMOTIONS)
    for emotion in EMOTIONS:
        rating_csv_path = f"ratings/{selected_img}.csv"
        remarks = f'{emotion}_EMOTION_RATING'
        display_rating_screen(emotion, rating_csv_path)

    stimulus_id, remarks = None, None


# FIXATION CROSS (END) -> 6s
image_frame = fixation_cross.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
screen.blit(fixation_cross, image_frame)

remarks = 'FIXATION CROSS'
pygame.display.update()
pygame.time.wait(6000)
remarks = None

# END OF PARADIGM
pygame.quit()


# SAVE EYE TRACKING DATA
my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
df_gaze = pd.DataFrame.from_dict(gaze_data)
df_gaze['seconds'] = df_gaze['device_time_stamp'].apply(lambda x: x / 1e6)
df_gaze.to_csv("eye_tracking_data.csv", index=False)
