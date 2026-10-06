"""
Face Recognition / Embedding Module
Extracts face embeddings and computes similarity.

Uses a lightweight approach based on:
1. OpenCV's DNN face recognizer (primary - no extra install)
2. Histogram-based fallback

This avoids requiring dlib/face_recognition compilation on Windows.
"""

import cv2
import numpy as np
import os
import urllib.request


# Directory for model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# OpenCV DNN face recognizer model files
FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
FACE_RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# OpenCV DNN face detector model files (YuNet - more reliable than Haar)
FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def ensure_model_downloaded(model_name, url):
    """Download model file if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        print(f"  From: {url}")
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"  Downloaded successfully to {model_path}")
        except Exception as e:
            print(f"  Failed to download: {e}")
            print(f"  Please download manually from: {url}")
            print(f"  Place it in: {MODEL_DIR}")
            return None
    
    return model_path


class FaceEmbedder:
    """
    Extracts 128-dimensional face embeddings using OpenCV's SFace model.
    Falls back to histogram-based features if model unavailable.
    """
    
    def __init__(self):
        self.recognizer = None
        self.yunet_detector = None
        self.use_sface = False
        self._init_sface()
    
    def _init_sface(self):
        """Initialize OpenCV SFace recognizer."""
        try:
            model_path = ensure_model_downloaded(FACE_RECOGNIZER_MODEL, FACE_RECOGNIZER_URL)
            if model_path and os.path.exists(model_path):
                self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")
                self.use_sface = True
                print("[INFO] SFace recognizer loaded successfully")
            else:
                print("[WARN] SFace model not available, using fallback")
        except Exception as e:
            print(f"[WARN] Could not load SFace: {e}")
            print("[INFO] Using histogram-based fallback")
    
    def _init_yunet(self, img_width, img_height):
        """Initialize YuNet detector for SFace alignment."""
        try:
            model_path = ensure_model_downloaded(FACE_DETECTOR_MODEL, FACE_DETECTOR_URL)
            if model_path and os.path.exists(model_path):
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model_path, "",
                    (img_width, img_height),
                    0.6,  # score threshold
                    0.3,  # nms threshold
                    5000  # top_k
                )
                return True
        except Exception as e:
            print(f"[WARN] Could not load YuNet: {e}")
        return False
    
    def extract_embedding(self, face_image):
        """
        Extract face embedding from a preprocessed face image.
        
        Args:
            face_image: Cropped and preprocessed face (BGR, any size)
            
        Returns:
            Numpy array of face embedding (128-d for SFace, 256-d for fallback)
        """
        if face_image is None or face_image.size == 0:
            return None
        
        if self.use_sface:
            return self._extract_sface_embedding(face_image)
        else:
            return self._extract_histogram_embedding(face_image)
    
    def _extract_sface_embedding(self, face_image):
        """Extract embedding using SFace model."""
        try:
            # Resize to SFace expected input: 112x112
            aligned = cv2.resize(face_image, (112, 112))
            
            # Try using YuNet for proper face alignment
            h, w = face_image.shape[:2]
            if self.yunet_detector is None:
                self._init_yunet(w, h)
            
            if self.yunet_detector is not None:
                self.yunet_detector.setInputSize((w, h))
                _, yunet_faces = self.yunet_detector.detect(face_image)
                
                if yunet_faces is not None and len(yunet_faces) > 0:
                    aligned = self.recognizer.alignCrop(face_image, yunet_faces[0])
            
            # Extract features
            embedding = self.recognizer.feature(aligned)
            return embedding.flatten()
        except Exception as e:
            print(f"[WARN] SFace extraction failed: {e}, using fallback")
            return self._extract_histogram_embedding(face_image)
    
    def _extract_histogram_embedding(self, face_image):
        """
        Fallback: Extract features using histogram-based approach.
        Creates a 256-dimensional feature vector from LBP-like histograms.
        """
        try:
            # Resize to standard size
            resized = cv2.resize(face_image, (128, 128))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
            
            # Divide face into 4x4 grid
            features = []
            cell_h, cell_w = gray.shape[0] // 4, gray.shape[1] // 4
            
            for i in range(4):
                for j in range(4):
                    cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    hist = cv2.calcHist([cell], [0], None, [16], [0, 256])
                    hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
                    features.extend(hist)
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Histogram extraction failed: {e}")
            return None
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Ensure same dimensionality
        if embedding1.shape != embedding2.shape:
            return 0.0
        
        # Cosine similarity
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot / (norm1 * norm2)
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, float(similarity)))
        return similarity
