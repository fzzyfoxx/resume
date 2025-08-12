from PIL import ImageFont
from importlib.resources import files

def preload_fonts():
    """
    Pre-load fonts into memory for reuse from the package's resources/fonts folder.
    Args:
        None
    Returns:
        dict: A dictionary where keys are font names and values are ImageFont objects.
    """
    fonts = {}
    try:
        # Access the resources/fonts folder within the package
        fonts_folder = files("geodoc_loader.resources").joinpath("fonts")
        for font_file in fonts_folder.iterdir():
            if font_file.suffix in ['.ttf', '.otf']:
                font_name = font_file.stem  # Extract font name without extension
                try:
                    fonts[font_name] = ImageFont.truetype(str(font_file))
                except Exception as e:
                    print(f"Error loading font '{font_name}' from '{font_file}': {e}")
    except Exception as e:
        print(f"Error accessing fonts folder: {e}")
    return fonts