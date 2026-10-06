"""
Face Detection Module
Detects faces in images using OpenCV's YuNet DNN detector.
Compatible with OpenCV 5.0+ (Haar Cascade removed in OpenCV 5).
"""

import cv2
import numpy as np
import os
import urllib.request


# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _ensure_model(model_name, url):
    """Download model file if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading {model_name}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[INFO] Downloaded to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {model_name}: {e}")
            return None
    return model_path


class FaceDetector:
    """Detects faces using OpenCV YuNet DNN detector."""
    
    def __init__(self):
        model_path = _ensure_model(FACE_DETECTOR_MODEL, FACE_DETECTOR_URL)
        if model_path is None:
            raise RuntimeError(
                f"Could not load face detector model. "
                f"Please download {FACE_DETECTOR_MODEL} from {FACE_DETECTOR_URL} "
                f"and place it in {MODEL_DIR}"
            )
        self._model_path = model_path
        # YuNet detector will be created per-image (needs input size)
        self._detector = None
    
    def _get_detector(self, width, height):
        """Create or reconfigure YuNet detector for given image size."""
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                self._model_path, "",
                (width, height),
                0.5,   # score threshold
                0.3,   # nms threshold
                5000   # top_k
            )
        else:
            self._detector.setInputSize((width, height))
        return self._detector
    
    def detect_faces(self, image):
        """
        Detect all faces in an image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            List of tuples: [(x, y, w, h, landmarks, score), ...]
            Each tuple contains bounding box, 5-point landmarks, and confidence.
            For backward compat, also works as (x, y, w, h) indexing.
        """
        if image is None or image.size == 0:
            return []
        
        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        
        _, raw_faces = detector.detect(image)
        
        if raw_faces is None:
            return []
        
        faces = []
        for face in raw_faces:
            x = int(face[0])
            y = int(face[1])
            fw = int(face[2])
            fh = int(face[3])
            score = float(face[-1])
            # Store as tuple: (x, y, w, h) for backward compatibility
            faces.append((x, y, fw, fh))
        
        return faces
    
    def detect_faces_raw(self, image):
        """
        Detect faces and return raw YuNet output (needed for SFace alignment).
        
        Returns:
            Raw numpy array from YuNet or None
        """
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        _, raw_faces = detector.detect(image)
        return raw_faces
    
    def detect_largest_face(self, image):
        """
        Detect the largest face in an image.
        
        Returns:
            (x, y, w, h) tuple or None
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])
    
    def crop_face(self, image, face_rect, margin=0.2):
        """
        Crop face from image with margin.
        
        Args:
            image: Input image
            face_rect: (x, y, w, h) tuple
            margin: Percentage margin around face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_rect[:4]
        img_h, img_w = image.shape[:2]
        
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        return cropped
    
    def draw_face_box(self, image, face_rect, label="", color=(0, 255, 0), thickness=2):
        """Draw bounding box around detected face."""
        result = image.copy()
        x, y, w, h = face_rect[:4]
        
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
            cv2.rectangle(
                result,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color, -1
            )
            cv2.putText(result, label, (x, y - 5), font, font_scale, (0, 0, 0), 2)
        
        return result
