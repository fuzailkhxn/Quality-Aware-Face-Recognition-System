"""
Face Quality Assessment Module
Assesses the quality of detected face images using multiple metrics:
- Sharpness (Laplacian variance)
- Brightness
- Contrast
- Face Size
- Illumination uniformity

Produces an overall quality score from 0-100.
"""

import cv2
import numpy as np


# Quality level thresholds (configurable)
QUALITY_THRESHOLDS = {
    "high": 80,     # 80-100 = High Quality
    "medium": 60,   # 60-79  = Medium Quality
    # 0-59 = Low Quality
}


class FaceQualityAssessor:
    """Assesses quality of face images using multiple metrics."""
    
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or QUALITY_THRESHOLDS
    
    def assess_sharpness(self, face_image):
        """
        Assess image sharpness using Laplacian variance.
        Higher variance = sharper image.
        
        Returns:
            Score 0-100
        """
        if face_image is None or face_image.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize: typical range 0-1000+, map to 0-100
        # Values below 50 are very blurry, above 500 are very sharp
        score = min(100, max(0, (laplacian_var / 500.0) * 100))
        return round(score, 1)
    
    def assess_brightness(self, face_image):
        """
        Assess image brightness.
        Optimal brightness is around 100-170 (on 0-255 scale).
        
        Returns:
            Score 0-100
        """
        if face_image is None or face_image.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        mean_brightness = np.mean(gray)
        
        # Optimal range: 100-170
        # Too dark (<50) or too bright (>220) = poor
        if 100 <= mean_brightness <= 170:
            score = 100
        elif mean_brightness < 100:
            score = max(0, (mean_brightness / 100.0) * 100)
        else:
            score = max(0, ((255 - mean_brightness) / 85.0) * 100)
        
        return round(min(100, score), 1)
    
    def assess_contrast(self, face_image):
        """
        Assess image contrast using standard deviation of pixel values.
        
        Returns:
            Score 0-100
        """
        if face_image is None or face_image.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        std_dev = np.std(gray)
        
        # Good contrast: std > 50, excellent: std > 70
        score = min(100, max(0, (std_dev / 70.0) * 100))
        return round(score, 1)
    
    def assess_face_size(self, face_image, min_size=80):
        """
        Assess if face is large enough for reliable recognition.
        
        Args:
            face_image: Cropped face image
            min_size: Minimum recommended face size in pixels
            
        Returns:
            Score 0-100
        """
        if face_image is None or face_image.size == 0:
            return 0
        
        h, w = face_image.shape[:2]
        face_size = min(h, w)
        
        if face_size >= min_size * 2:
            score = 100
        elif face_size >= min_size:
            score = 60 + (face_size - min_size) / min_size * 40
        else:
            score = max(0, (face_size / min_size) * 60)
        
        return round(min(100, score), 1)
    
    def assess_illumination(self, face_image):
        """
        Assess illumination uniformity.
        Splits face into left/right halves and compares brightness.
        
        Returns:
            Score 0-100
        """
        if face_image is None or face_image.size == 0:
            return 0
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        h, w = gray.shape
        
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        
        # Calculate asymmetry
        if max(left_mean, right_mean) == 0:
            return 0
        
        asymmetry = abs(left_mean - right_mean) / max(left_mean, right_mean)
        
        # Low asymmetry = good illumination
        score = max(0, (1 - asymmetry * 3) * 100)
        return round(min(100, score), 1)
    
    def assess_quality(self, face_image):
        """
        Perform complete quality assessment on a face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Dictionary with quality metrics and overall score
        """
        if face_image is None or face_image.size == 0:
            return {
                "overall_score": 0,
                "quality_level": "Low",
                "sharpness": 0,
                "brightness": 0,
                "contrast": 0,
                "face_size": 0,
                "illumination": 0,
                "details": {}
            }
        
        # Calculate individual metrics
        sharpness = self.assess_sharpness(face_image)
        brightness = self.assess_brightness(face_image)
        contrast = self.assess_contrast(face_image)
        face_size = self.assess_face_size(face_image)
        illumination = self.assess_illumination(face_image)
        
        # Weighted average for overall score
        weights = {
            "sharpness": 0.30,
            "brightness": 0.20,
            "contrast": 0.20,
            "face_size": 0.15,
            "illumination": 0.15
        }
        
        overall = (
            sharpness * weights["sharpness"] +
            brightness * weights["brightness"] +
            contrast * weights["contrast"] +
            face_size * weights["face_size"] +
            illumination * weights["illumination"]
        )
        overall = round(min(100, overall), 1)
        
        # Determine quality level
        if overall >= self.thresholds["high"]:
            quality_level = "High"
        elif overall >= self.thresholds["medium"]:
            quality_level = "Medium"
        else:
            quality_level = "Low"
        
        def metric_label(score):
            if score >= 80:
                return "Good"
            elif score >= 60:
                return "Fair"
            else:
                return "Poor"
        
        return {
            "overall_score": overall,
            "quality_level": quality_level,
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "face_size": face_size,
            "illumination": illumination,
            "details": {
                "sharpness_label": metric_label(sharpness),
                "brightness_label": metric_label(brightness),
                "contrast_label": metric_label(contrast),
                "face_size_label": metric_label(face_size),
                "illumination_label": metric_label(illumination),
            }
        }
