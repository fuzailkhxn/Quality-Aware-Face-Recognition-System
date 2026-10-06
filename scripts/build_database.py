"""
Build Face Database Script

Scans the dataset/registered/ directory for face images,
detects faces, extracts embeddings, and saves the database.

Usage:
    python scripts/build_database.py

Directory structure expected:
    dataset/registered/
        PersonName1/
            image1.jpg
            image2.jpg
        PersonName2/
            image1.jpg
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
from src.face_detector import FaceDetector
from src.quality_assessment import FaceQualityAssessor
from src.preprocessing import FacePreprocessor
from src.face_recognizer import FaceEmbedder
from src.database import FaceDatabase, DEFAULT_REGISTERED_DIR


def build_database(registered_dir=None):
    """Build face embeddings database from registered images."""
    
    registered_dir = registered_dir or DEFAULT_REGISTERED_DIR
    
    print("=" * 60)
    print("  Face Database Builder")
    print("  Quality-Aware Face Recognition System")
    print("=" * 60)
    print()
    
    if not os.path.exists(registered_dir):
        print(f"[ERROR] Registered faces directory not found: {registered_dir}")
        print(f"\nPlease create the directory and add face images:")
        print(f"  {registered_dir}/")
        print(f"    PersonName1/")
        print(f"      image1.jpg")
        print(f"      image2.jpg")
        print(f"    PersonName2/")
        print(f"      image1.jpg")
        os.makedirs(registered_dir, exist_ok=True)
        print(f"\n[INFO] Created empty directory: {registered_dir}")
        return
    
    # Initialize components
    print("[INFO] Initializing face recognition components...")
    detector = FaceDetector()
    quality_assessor = FaceQualityAssessor()
    preprocessor = FacePreprocessor()
    embedder = FaceEmbedder()
    database = FaceDatabase()
    
    # Clear existing database
    database.database = {}
    
    # Scan directory
    persons = [d for d in os.listdir(registered_dir)
               if os.path.isdir(os.path.join(registered_dir, d))]
    
    if not persons:
        print(f"\n[WARN] No person directories found in {registered_dir}")
        print("Please add subdirectories with face images.")
        return
    
    print(f"\nFound {len(persons)} identities")
    print("-" * 40)
    
    total_images = 0
    total_embeddings = 0
    
    for person_name in sorted(persons):
        person_dir = os.path.join(registered_dir, person_name)
        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        
        print(f"\n{person_name} -> {len(image_files)} images")
        total_images += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  [SKIP] Could not load: {img_file}")
                    continue
                
                # Detect face
                face_rect = detector.detect_largest_face(image)
                if face_rect is None:
                    print(f"  [SKIP] No face detected in: {img_file}")
                    continue
                
                # Crop face
                face_crop = detector.crop_face(image, face_rect, margin=0.3)
                
                # Assess quality
                quality = quality_assessor.assess_quality(face_crop)
                
                # Preprocess
                processed, _ = preprocessor.preprocess_based_on_quality(face_crop, quality)
                
                # Extract embedding
                embedding = embedder.extract_embedding(processed)
                
                if embedding is not None:
                    database.add_embedding(person_name, embedding)
                    total_embeddings += 1
                    print(f"  [OK] {img_file} - Quality: {quality['overall_score']:.0f}/100 ({quality['quality_level']})")
                else:
                    print(f"  [SKIP] Could not extract embedding: {img_file}")
            
            except Exception as e:
                print(f"  [ERROR] {img_file}: {e}")
    
    print("\n" + "=" * 60)
    
    if total_embeddings > 0:
        database.save()
        print(f"\n[SUCCESS] Database created successfully!")
        print(f"   Total identities: {len(database.get_all_identities())}")
        print(f"   Total images processed: {total_images}")
        print(f"   Total embeddings stored: {total_embeddings}")
        print(f"\n   Database saved to: {database.embeddings_file}")
    else:
        print(f"\n[WARNING] No embeddings were extracted.")
        print(f"   Make sure your images contain clearly visible faces.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    build_database()
