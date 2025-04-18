from PIL import Image, ImageDraw, ImageFont
from os import path

class ImageOverlay:
    """
    A class to handle text overlay operations on images.
    
    Parameters:
        image_file_name (str): Path to the image file
        font (str): Path to the font file
        bg_tint_color (tuple, optional): RGB color tuple for the background tint
        bg_transparency (float, optional): Background transparency value (0-1)
        text_color (tuple, optional): RGBA color tuple for the text
        draw_shadow (bool, optional): Whether to draw text shadow
        shadow_color (tuple, optional): RGBA color tuple for the shadow
        shadow_offset (tuple, optional): (x,y) offset for the shadow
    """
    def __init__(self, image_file_name, font, bg_tint_color=(0, 0, 0), bg_transparency=0, text_color=None, draw_shadow=False, shadow_color=(0, 0, 0, 128), shadow_offset=(5, 5)):
        self.image_file_name = image_file_name
        self.font = font
        self.bg_tint_color = bg_tint_color
        self.bg_transparency = bg_transparency
        self.text_color = text_color
        self.draw_shadow = draw_shadow
        self.shadow_color = shadow_color 
        self.shadow_offset = shadow_offset
        self.image = None

    def load_image(self):
        """
        Loads and converts the image file to RGBA format.
        
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        if not path.exists(self.image_file_name):
            print("File not found: " + self.image_file_name)
            return False
        self.image = Image.open(self.image_file_name).convert("RGBA")
        return True

    def overlay_text(self, text_positions):
        """
        Overlays text on the image at specified positions.
        
        Parameters:
            text_positions (list): List of tuples containing (x_ratio, y_ratio, text)
                where x_ratio and y_ratio are float values between 0-1 representing
                the relative position on the image, and text is the string to display
                
        Returns:
            tuple: (bool, str) - (success status, output filename if successful or error message)
        """
        if self.image is None:
            return False, "Image not loaded."

        try:
            overlay = Image.new('RGBA', self.image.size, self.bg_tint_color + (0,))
            draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.

            image_width, image_height = self.image.size

            # Calculate font size based on image height
            font_size = int(image_height * 0.05)  # 5% of the image height
            font = ImageFont.truetype(self.font, font_size)

            for position in text_positions:
                x_ratio, y_ratio, text = position  # Unpack the position tuple
                x = int(x_ratio * image_width)  # Calculate absolute x position
                y = int(y_ratio * image_height)  # Calculate absolute y position

                # Calculate text dimensions
                text_width = draw.textlength(text=text, font=font)
                text_height = font_size  # Use the calculated font size for height

                # Center the text
                centered_x = x - (text_width // 2)
                centered_y = y - (text_height // 2)

                # Draw shadow if enabled
                if self.draw_shadow:
                    shadow_x = centered_x + self.shadow_offset[0]
                    shadow_y = centered_y + self.shadow_offset[1]
                    draw.text((shadow_x, shadow_y), text, fill=self.shadow_color, font=font)

                # Draw a rectangle behind the text
                draw.rectangle((centered_x, centered_y, centered_x + text_width, centered_y + text_height), fill=self.bg_tint_color + (int(255 * self.bg_transparency),))

                # Draw the text with the specified text color
                draw.text((centered_x, centered_y), text, fill=self.text_color, font=font)

            img = Image.alpha_composite(self.image, overlay)
            output_file_name = path.splitext(self.image_file_name)[0] + "_overlay.png"
            img.save(output_file_name)
            print(f"Overlay image saved as: {output_file_name}")
            return True, output_file_name
            
        except Exception as e:
            error_msg = f"Error during overlay: {str(e)}"
            print(error_msg)
            return False, error_msg

if __name__ == "__main__":

    image = r"frames\frame_000000.jpg"
    font = r"fonts\LoveDays-2v7Oe.ttf"

    bg_tint_color = (0, 0, 0)
    bg_transparency = 0  # Degree of transparency of background, 0 is fully transparent and 1 is fully opaque
    alpha = 1  # Degree of transparency of text, 0-1
    alpha = int(alpha * 255)
    text_color = (122, 245, 248, alpha)  # Text color (RGBA)
    text_positions = [
        (0.2, 0.15, "Skbidi Ohio Rizzler"),  # (x_ratio, y_ratio, text)
        (0.8, 0.25, "Boat goes binted")   # Add more text positions as needed
    ]

    # Create an instance of ImageOverlay and perform the overlay
    overlay_instance = ImageOverlay(image, font, bg_tint_color, bg_transparency, text_color, draw_shadow=True)
    if overlay_instance.load_image():
        success, result = overlay_instance.overlay_text(text_positions)
        if not success:
            print(f"Failed to create overlay: {result}")

