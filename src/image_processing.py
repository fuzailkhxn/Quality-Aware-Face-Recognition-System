import cv2
import numpy as np


def load_image(image_path):
    """
    Load image from given path
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Image not found")

    return image


def resize_image(image, size=(224, 224)):
    """
    Resize image for model input
    """
    resized = cv2.resize(image, size)
    return resized


def preprocess_image(image):
    """
    Preprocess image:
    - Convert BGR to RGB
    - Normalize pixel values
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0

    return image


def detect_face(image):
    """
    Detect face using OpenCV Haar Cascade
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 
        "haarcascade_frontalface_default.xml"
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return faces


def process_image(image_path):
    """
    Complete image processing pipeline
    """

    image = load_image(image_path)

    image = resize_image(image)

    processed_image = preprocess_image(image)

    faces = detect_face(processed_image)

    return processed_image, faces