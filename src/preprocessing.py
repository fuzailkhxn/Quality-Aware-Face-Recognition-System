"""
Face Preprocessing Module
Applies quality-aware preprocessing to face images.
Enhances low-quality images before recognition.
"""

import cv2
import numpy as np


class FacePreprocessor:
    """Applies quality-aware preprocessing to face images."""
    
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
    
    def enhance_brightness(self, image, target_mean=127):
        """
        Adjust image brightness towards a target mean.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        current_mean = np.mean(gray)
        
        if current_mean == 0:
            return image
        
        factor = target_mean / current_mean
        factor = np.clip(factor, 0.5, 2.0)  # Limit adjustment
        
        enhanced = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return enhanced
    
    def enhance_contrast(self, image):
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def enhance_sharpness(self, image):
        """
        Apply mild sharpening filter.
        """
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def denoise(self, image):
        """
        Apply mild denoising.
        """
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        return denoised
    
    def resize_face(self, image):
        """
        Resize face to target size for embedding extraction.
        """
        return cv2.resize(image, self.target_size)
    
    def preprocess_based_on_quality(self, face_image, quality_result):
        """
        Apply quality-aware preprocessing.
        
        - High quality: minimal processing (just resize)
        - Medium quality: moderate enhancement
        - Low quality: aggressive enhancement
        
        Args:
            face_image: Cropped face image
            quality_result: Dict from FaceQualityAssessor.assess_quality()
            
        Returns:
            Preprocessed face image, list of applied enhancements
        """
        processed = face_image.copy()
        enhancements = []
        quality_level = quality_result["quality_level"]
        
        if quality_level == "High":
            # Minimal processing
            enhancements.append("Standard normalization")
        
        elif quality_level == "Medium":
            # Moderate enhancement
            if quality_result["brightness"] < 60:
                processed = self.enhance_brightness(processed)
                enhancements.append("Brightness enhanced")
            
            if quality_result["contrast"] < 60:
                processed = self.enhance_contrast(processed)
                enhancements.append("Contrast enhanced (CLAHE)")
            
            if not enhancements:
                enhancements.append("Minor adjustments")
        
        else:  # Low quality
            # Aggressive enhancement
            if quality_result["brightness"] < 60:
                processed = self.enhance_brightness(processed)
                enhancements.append("Brightness enhanced")
            
            processed = self.enhance_contrast(processed)
            enhancements.append("Contrast enhanced (CLAHE)")
            
            if quality_result["sharpness"] < 40:
                processed = self.enhance_sharpness(processed)
                enhancements.append("Sharpening applied")
            
            # Denoise for very low quality
            if quality_result["overall_score"] < 30:
                processed = self.denoise(processed)
                enhancements.append("Denoising applied")
        
        # Always resize to target
        processed = self.resize_face(processed)
        
        return processed, enhancements
