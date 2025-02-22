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
    # Calculate true percentage
    detection_events = get_detection_events()
    total_events = len(detection_events)
    true_count = sum(1 for _, event in detection_events if event)
    true_percentage = (true_count / total_events * 100) if total_events > 0 else 0

    # Compute cumulative score over time
    times = []
    cumulative = []
    score = 0
    for t, event in detection_events:
        times.append(t)
        if event:
            score += 1
        cumulative.append(score)

    # Create a matplotlib graph: Time vs. Cumulative Score
    plt.figure(figsize=(6, 4))
    plt.plot(times, cumulative, marker='o')
    plt.xlabel('Time (ms)')
    plt.ylabel('Cumulative Score')
    theoretical_upper = total_events
    plt.ylim(bottom=0, top=theoretical_upper)
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
        screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Statistics Display")
    screen_width, screen_height = screen.get_size()

    # Scale the image to fit the screen
    graph_image = scale_image_to_screen(graph_image, screen_width//2, screen_height//2)


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

        # Render and display the percentage text
        percentage_text = font.render(f"True Percentage: {true_percentage:.2f}%", True, (0, 0, 0))
        screen.blit(percentage_text, (20, 20))

        # Blit the graph image (you can adjust the position as needed)
        screen.blit(graph_image, (20, 70))

        # Update the display
        pygame.display.flip()
        clock.tick(30)  # Limit the loop to 30 FPS

    # Clean up when exiting the statistics display
    pygame.quit()