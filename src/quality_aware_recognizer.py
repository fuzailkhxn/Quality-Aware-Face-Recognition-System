"""
Quality-Aware Adaptive Recognizer

This is the core module of the project.
It integrates face detection, quality assessment, preprocessing,
and recognition into a single quality-aware pipeline.

Key concept: The recognition decision adapts based on the assessed
image quality. Low-quality images are treated with more conservative
thresholds, while high-quality images allow more confident decisions.
"""

import cv2
import numpy as np

from .face_detector import FaceDetector
from .quality_assessment import FaceQualityAssessor
from .preprocessing import FacePreprocessor
from .face_recognizer import FaceEmbedder
from .database import FaceDatabase


# Adaptive recognition thresholds based on quality
ADAPTIVE_THRESHOLDS = {
    "High": {
        "recognition_threshold": 0.45,   # More lenient - trust the embedding
        "confidence_multiplier": 1.0,    # Full confidence
        "label": "High Quality - Full Confidence"
    },
    "Medium": {
        "recognition_threshold": 0.52,   # Slightly stricter
        "confidence_multiplier": 0.85,   # Reduced confidence
        "label": "Medium Quality - Moderate Confidence"
    },
    "Low": {
        "recognition_threshold": 0.60,   # Much stricter
        "confidence_multiplier": 0.65,   # Significantly reduced confidence
        "label": "Low Quality - Recognition Less Reliable"
    }
}


class QualityAwareRecognizer:
    """
    Performs quality-aware face recognition.
    
    Pipeline:
    1. Detect face in image
    2. Crop face region
    3. Assess face quality
    4. Apply quality-based preprocessing
    5. Extract face embedding
    6. Compare with registered faces
    7. Apply quality-aware confidence adjustment
    8. Make final adaptive decision
    """
    
    def __init__(self, database=None, thresholds=None):
        self.detector = FaceDetector()
        self.quality_assessor = FaceQualityAssessor()
        self.preprocessor = FacePreprocessor()
        self.embedder = FaceEmbedder()
        self.database = database or FaceDatabase()
        self.thresholds = thresholds or ADAPTIVE_THRESHOLDS
    
    def recognize(self, image):
        """
        Perform full quality-aware recognition on an image.
        
        Args:
            image: Input image (BGR format, numpy array)
            
        Returns:
            Dictionary with complete recognition results
        """
        result = {
            "success": False,
            "face_detected": False,
            "face_count": 0,
            "face_rect": None,
            "face_crop": None,
            "quality": None,
            "enhancements": [],
            "person": "Unknown",
            "raw_similarity": 0.0,
            "adjusted_confidence": 0.0,
            "quality_level": "Unknown",
            "adaptive_threshold": 0.0,
            "decision": "No face detected",
            "decision_details": "",
            "annotated_image": None,
            "all_similarities": {},
            "message": ""
        }
        
        if image is None:
            result["message"] = "Invalid image provided."
            return result
        
        # Step 1: Detect faces
        faces = self.detector.detect_faces(image)
        result["face_count"] = len(faces)
        
        if len(faces) == 0:
            result["message"] = "No face detected. Please upload a clearer face image."
            result["annotated_image"] = image.copy()
            return result
        
        result["face_detected"] = True
        
        if len(faces) > 1:
            result["message"] = f"Multiple faces detected ({len(faces)}). Using the largest face."
        
        # Use the largest face
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        result["face_rect"] = face_rect
        
        # Step 2: Crop face
        face_crop = self.detector.crop_face(image, face_rect, margin=0.3)
        result["face_crop"] = face_crop
        
        if face_crop is None or face_crop.size == 0:
            result["message"] = "Could not crop face region."
            return result
        
        # Step 3: Assess quality
        quality = self.quality_assessor.assess_quality(face_crop)
        result["quality"] = quality
        result["quality_level"] = quality["quality_level"]
        
        # Step 4: Quality-aware preprocessing
        processed_face, enhancements = self.preprocessor.preprocess_based_on_quality(
            face_crop, quality
        )
        result["enhancements"] = enhancements
        
        # Step 5: Extract embedding
        embedding = self.embedder.extract_embedding(processed_face)
        
        if embedding is None:
            result["message"] = "Could not extract face features."
            return result
        
        # Step 6: Compare with database
        if self.database.is_empty():
            result["message"] = "No registered identities found. Please add images to dataset/registered/."
            # Still mark as successful face detection + quality assessment
            result["success"] = True
            result["decision"] = "No database - Cannot recognize"
            result["annotated_image"] = self._annotate_image(
                image, face_rect, result
            )
            return result
        
        best_match, best_similarity, all_similarities = self._find_best_match(embedding)
        result["all_similarities"] = all_similarities
        result["raw_similarity"] = best_similarity
        
        # Step 7: Quality-aware confidence adjustment
        quality_level = quality["quality_level"]
        threshold_config = self.thresholds[quality_level]
        adaptive_threshold = threshold_config["recognition_threshold"]
        confidence_multiplier = threshold_config["confidence_multiplier"]
        
        result["adaptive_threshold"] = adaptive_threshold
        
        # Adjust confidence based on quality
        adjusted_confidence = best_similarity * confidence_multiplier
        result["adjusted_confidence"] = adjusted_confidence
        
        # Step 8: Make final decision
        if best_similarity >= adaptive_threshold:
            result["person"] = best_match
            result["decision"] = "RECOGNIZED"
            result["success"] = True
            
            if quality_level == "Low":
                result["decision_details"] = (
                    f"LOW QUALITY — Recognition Less Reliable. "
                    f"Adaptive threshold applied: {adaptive_threshold:.2f}"
                )
            elif quality_level == "Medium":
                result["decision_details"] = (
                    f"Medium quality — Moderate confidence. "
                    f"Adaptive threshold: {adaptive_threshold:.2f}"
                )
            else:
                result["decision_details"] = (
                    f"High quality — Full confidence recognition."
                )
        else:
            result["person"] = "Unknown"
            result["decision"] = "UNKNOWN PERSON"
            result["success"] = True
            
            if best_similarity > 0.3:
                result["decision_details"] = (
                    f"Closest match: {best_match} (similarity: {best_similarity:.2f}), "
                    f"but below adaptive threshold ({adaptive_threshold:.2f}) "
                    f"for {quality_level} quality."
                )
            else:
                result["decision_details"] = "No close match found in database."
        
        # Annotate image
        result["annotated_image"] = self._annotate_image(
            image, face_rect, result
        )
        
        if not result["message"]:
            result["message"] = threshold_config["label"]
        
        return result
    
    def _find_best_match(self, embedding):
        """
        Find the best matching identity in the database.
        
        Returns:
            (best_name, best_similarity, all_similarities_dict)
        """
        best_match = "Unknown"
        best_similarity = 0.0
        all_similarities = {}
        
        for name in self.database.get_all_identities():
            person_embeddings = self.database.get_embeddings(name)
            
            if not person_embeddings:
                continue
            
            # Compare with each registered embedding and take the best
            similarities = []
            for reg_emb in person_embeddings:
                sim = self.embedder.compute_similarity(embedding, reg_emb)
                similarities.append(sim)
            
            # Use average of top similarities for robustness
            if similarities:
                similarities.sort(reverse=True)
                # Average of top matches (or all if few)
                top_k = min(3, len(similarities))
                avg_sim = np.mean(similarities[:top_k])
                all_similarities[name] = round(float(avg_sim), 4)
                
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_match = name
        
        return best_match, round(float(best_similarity), 4), all_similarities
    
    def _annotate_image(self, image, face_rect, result):
        """
        Draw annotations on the image.
        """
        annotated = image.copy()
        x, y, w, h = face_rect
        
        # Choose color based on decision
        if result["decision"] == "RECOGNIZED":
            color = (0, 255, 0)  # Green
        elif result["decision"] == "UNKNOWN PERSON":
            color = (0, 255, 0) if False else (0, 0, 255)  # Red
        else:
            color = (255, 165, 0)  # Orange
        
        # Draw face rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{result['person']} ({result['adjusted_confidence']:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(
            annotated,
            (x, y - label_size[1] - 10),
            (x + label_size[0] + 5, y),
            color, -1
        )
        cv2.putText(
            annotated, label,
            (x + 2, y - 5),
            font, font_scale, (255, 255, 255), thickness
        )
        
        # Quality label
        quality_label = f"Quality: {result['quality']['overall_score']:.0f}/100 ({result['quality_level']})"
        cv2.putText(
            annotated, quality_label,
            (x, y + h + 20),
            font, 0.5, color, 1
        )
        
        return annotated
