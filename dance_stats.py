import matplotlib.pyplot as plt
import io
import pygame
from shared import *
def scale_image_to_screen(image, screen_width, screen_height):
    """
    Scales the input image to fit within the screen dimensions while maintaining aspect ratio.
    """
    image_width, image_height = image.get_size()
    aspect_ratio = image_width / image_height

    if image_width > screen_width or image_height > screen_height:
        if image_width / screen_width > image_height / screen_height:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
        image = pygame.transform.scale(image, (new_width, new_height))
    
    return image

def display_statistics():
    # Get detection events
    detection_events = get_detection_events()
    total_events = len(detection_events)

    # Count occurrences of GREAT, GOOD, and BAD
    great_count = sum(1 for _, event in detection_events if event == "GREAT")
    good_count = sum(1 for _, event in detection_events if event == "OK")
    bad_count = sum(1 for _, event in detection_events if event == "BAD")

    # Compute percentages
    great_percentage = (great_count / total_events * 100) if total_events > 0 else 0
    good_percentage = (good_count / total_events * 100) if total_events > 0 else 0
    bad_percentage = (bad_count / total_events * 100) if total_events > 0 else 0

    # Compute cumulative score over time
    times = []
    scores = []
    cumulative_score = 0

    for t, event in detection_events:
        times.append(t)
        if event == "GREAT":
            cumulative_score += 10
        elif event == "OK":
            cumulative_score += 5
        scores.append(cumulative_score)

    # Create a matplotlib graph with color-coded segments
    plt.figure(figsize=(6, 4))

    for i in range(0, len(times)-1):
        color = "red"  # Default to BAD
        if detection_events[i+1][1] == "GREAT":
            color = "green"
        elif detection_events[i+1][1] == "OK":
            color = "gold"

        plt.plot(times[i:i+2], scores[i:i+2], marker='o', color=color, linewidth=2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Cumulative Score')
    upper_limit = total_events * 10
    plt.ylim(bottom=0, top=upper_limit)
    plt.title('Time vs. Cumulative Score')
    plt.tight_layout()

    # Save the plot to an in-memory bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG')
    buffer.seek(0)
    plt.close()

    # Load the image from the buffer into Pygame
    graph_image = pygame.image.load(buffer, 'graph.png')

    # Set up Pygame display (if not already set)
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("Statistics Display")
    screen_width, screen_height = screen.get_size()

    # Scale the image to fit the screen
    graph_image = scale_image_to_screen(graph_image, screen_width // 2, screen_height // 2)

    # Set up font for text display
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    running = True
    while running:
        # Process events so the window remains responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen with a white background
        screen.fill((255, 255, 255))

        # Render and display percentages
        percentage_text_great = font.render(f"GREAT: {great_percentage:.2f}%", True, (0, 200, 0))
        percentage_text_good = font.render(f"OK: {good_percentage:.2f}%", True, (200, 200, 0))
        percentage_text_bad = font.render(f"BAD: {bad_percentage:.2f}%", True, (200, 0, 0))

        screen.blit(percentage_text_great, (20, 20))
        screen.blit(percentage_text_good, (20, 60))
        screen.blit(percentage_text_bad, (20, 100))

        # Blit the graph image
        screen.blit(graph_image, (20, 150))

        # Update the display
        pygame.display.flip()
        clock.tick(30)  # Limit the loop to 30 FPS

    pygame.quit()
