import pygame
import os

def selectpage(screen, width, height, base_folder="song"):
    """Displays a scrollable list of folders inside `base_folder` and allows selection."""

    # Constants

    FONT_SIZE = 30
    ITEM_HEIGHT = 40  # Spacing between items
    SCROLL_SPEED = 20  # How much to scroll per step

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # Screen setup
    # Font setup
    font = pygame.font.Font(None, FONT_SIZE)

    # Get list of folders in the specified directory
    if not os.path.exists(base_folder):
        print(f"Error: Directory '{base_folder}' does not exist.")
        return None
    
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folder_paths = [os.path.join(base_folder, f) for f in folders]

    # Scroll settings
    scroll_y = 0  # Vertical scroll offset

    def draw_folders():
        """Draws the folder list with scrolling."""
        screen.fill(WHITE)
        
        for i, folder in enumerate(folders):
            y_pos = i * ITEM_HEIGHT - scroll_y
            if 0 <= y_pos < height:  # Only draw visible items
                text_surface = font.render(folder, True, BLACK)
                screen.blit(text_surface, (50, y_pos + 10))

        pygame.display.flip()

    def get_clicked_folder(mouse_pos):
        """Returns the folder path if an item is clicked."""
        x, y = mouse_pos
        index = (y + scroll_y) // ITEM_HEIGHT
        if 0 <= index < len(folder_paths):
            return folder_paths[index]
        return None

    # Main loop
    running = True
    selected_folder = None

    while running:
        draw_folders()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse scroll handling
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    scroll_y = max(scroll_y - SCROLL_SPEED, 0)
                elif event.button == 5:  # Scroll down
                    max_scroll = max(0, len(folders) * ITEM_HEIGHT - height)
                    scroll_y = min(scroll_y + SCROLL_SPEED, max_scroll)
                elif event.button == 1:  # Left click
                    selected_folder = get_clicked_folder(event.pos)
                    if selected_folder:
                        running = False  # Exit the loop on selection

            # Keyboard scrolling
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    scroll_y = max(scroll_y - SCROLL_SPEED, 0)
                elif event.key == pygame.K_DOWN:
                    max_scroll = max(0, len(folders) * ITEM_HEIGHT - height)
                    scroll_y = min(scroll_y + SCROLL_SPEED, max_scroll)


    return selected_folder  # Return the selected folder path

# Example usage:


