def normalize_image(image):
    """Normalize a single image array to the 0-1 range."""
    return image.astype('float32') / 255.0