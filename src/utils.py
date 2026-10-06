"""
Utility functions for the face recognition system.
"""

import cv2
import numpy as np


def apply_blur(image, kernel_size=15):
    """Apply Gaussian blur to simulate poor quality."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_low_brightness(image, factor=0.3):
    """Reduce brightness to simulate dark images."""
    dark = (image.astype(np.float32) * factor).astype(np.uint8)
    return dark


def apply_low_contrast(image, factor=0.3):
    """Reduce contrast."""
    mean = np.mean(image)
    low_contrast = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(low_contrast, 0, 255).astype(np.uint8)


def apply_noise(image, intensity=50):
    """Add Gaussian noise."""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def load_image_from_upload(uploaded_file):
    """
    Convert a Streamlit uploaded file to OpenCV image.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        BGR numpy array or None
    """
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        uploaded_file.seek(0)  # Reset file pointer
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def bgr_to_rgb(image):
    """Convert BGR to RGB for display."""
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    """Convert RGB to BGR for OpenCV processing."""
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
