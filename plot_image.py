from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def plot_points_on_image(image_path, points):
    """
    Plots points on an image and displays the result.

    Args:
        image_path: Path to the image file.
        points: A list of points, where each point is a list [x, y].
    """
    try:
        # Open the image using Pillow
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Define point properties (color, size)
        point_color = (255, 0, 0)  # Red color
        point_size = 5

        # Plot each point
        for x, y in points:
            # Calculate bounding box for the point
            x1 = x - point_size
            y1 = y - point_size
            x2 = x + point_size
            y2 = y + point_size
            draw.ellipse([(x1, y1), (x2, y2)], fill=point_color)

        # Convert PIL Image to matplotlib format for display
        plt.imshow(img)

        # Plot points again using matplotlib for better visual integration (optional, but can improve point visibility)
        for x, y in points:
            plt.plot(x, y, marker='o', markersize=point_size/2, markeredgecolor='white', markerfacecolor='red') # Adjust markersize as needed

        plt.title('Image with Plotted Points')
        plt.axis('on') # or 'off' to hide axes
        plt.show()

    except FileNotFoundError:
        print(f"Error: Image file not found at path: {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    image_file_path = "C:\\Users\\lfh20\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-02-22 132413.png"  # Replace with the actual path to your image file
    points_to_plot = [[1850, 100], [1700, 800]]  # Example points [x1, y1], [x2, y2]

    plot_points_on_image(image_file_path, points_to_plot)