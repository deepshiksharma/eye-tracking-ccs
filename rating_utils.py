import pygame
import os

def get_points(screen_height, num_points, hover_radius, gap):
    line_y = int(screen_height * 0.60)
    points = []
    for i in range(num_points):
        x = gap * (i + 1)
        y = line_y
        points.append(pygame.Rect(x - hover_radius, y - hover_radius, hover_radius * 2, hover_radius * 2))
    return points

def load_emoji_images(icon_dir, names, size=(100, 100)):
    images = []
    for name in names:
        path = os.path.join(icon_dir, f"{name}.png")
        try:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, size)
            images.append(img)
        except:
            images.append(None)
    return images

def display_rating_screen(emotion, screen, config, screen_width, screen_height, clock):
    FPS = 60
    font = pygame.font.SysFont(None, 64)
    label_font = pygame.font.SysFont(None, 36)
    bg_color = pygame.Color('black')
    
    base_radius = 25
    hover_radius = 30
    num_points = 9
    gap = screen_width // (num_points + 1)
    line_y = int(screen_height * 0.60)

    selected_rating = None
    points = get_points(screen_height, num_points, hover_radius, gap)
    running = True

    rating_labels = [
        emotion.capitalize() if lbl is None else lbl
        for lbl in config['ratings']['rating_labels_template']
    ]
    emoji_names = config['ratings']['emotion_rating_image_order'][emotion]
    emoji_images = load_emoji_images(config['ratings']['icon_dir'], emoji_names)

    while running:
        screen.fill(bg_color)
        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True
        
        text_line_1 = font.render(f"On a scale of 1 to 9, rate how much {emotion.upper()} did you feel ", True, (255, 255, 255))
        text_line_2 = font.render(f"after viewing the image", True, (255, 255, 255))
        screen.blit(text_line_1, ((screen_width - text_line_1.get_width()) // 2, 175))
        screen.blit(text_line_2, ((screen_width - text_line_2.get_width()) // 2, 250))

        pygame.draw.line(screen, (255, 255, 255), (gap, line_y), (screen_width - gap, line_y), 2)

        for i, rect in enumerate(points):
            is_hovered = rect.collidepoint(mouse_pos)
            point_x = rect.centerx
            point_y = rect.centery
            radius = hover_radius if is_hovered else base_radius
            color = (255, 255, 255)
            pygame.draw.circle(screen, color, (point_x, point_y), radius)
            number_label = label_font.render(str(i + 1), True, color)
            screen.blit(number_label, (point_x - number_label.get_width() // 2, line_y + 40))
            
            if is_hovered and click and selected_rating is None:
                selected_rating = i + 1
                running = False

        emoji_positions = [0, 2, 4, 6, 8]
        for idx, point_index in enumerate(emoji_positions):
            if idx < len(emoji_images) and emoji_images[idx]:
                img = emoji_images[idx]
                x = points[point_index].centerx
                y = line_y - img.get_height() - 80
                screen.blit(img, (x - img.get_width() // 2, y))

        for idx, point_index in enumerate(emoji_positions):
            label_text = rating_labels[idx]
            label = label_font.render(label_text, True, (200, 200, 200))
            x = points[point_index].centerx
            y = line_y + 90
            screen.blit(label, (x - label.get_width() // 2, y))

        pygame.mouse.set_visible(True)
        pygame.display.flip()
        clock.tick(FPS)
        pygame.mouse.set_visible(False)

    return selected_rating
