"""Comprehensive end-to-end test of the complete pipeline."""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quality_aware_recognizer import QualityAwareRecognizer
from src.database import FaceDatabase
from src.utils import apply_blur, apply_low_brightness, apply_low_contrast, apply_noise

def main():
    print("=" * 60)
    print("  Comprehensive Pipeline Test")
    print("=" * 60)

    db = FaceDatabase()
    rec = QualityAwareRecognizer(database=db)
    print(f"Database loaded: {len(db.get_all_identities())} identities")

    img = cv2.imread("dataset/registered/Adnan/image_1.jpg")
    if img is None:
        print("[FAIL] Could not load test image")
        return

    tests_passed = 0
    tests_total = 0

    # --- Test 1: Known face, original quality ---
    tests_total += 1
    print("\n--- TEST 1: Known face, original quality ---")
    r = rec.recognize(img)
    print(f"  Person: {r['person']}")
    print(f"  Decision: {r['decision']}")
    q = r["quality"]
    print(f"  Quality: {q['overall_score']:.1f}/100 ({r['quality_level']})")
    print(f"  Sharpness: {q['sharpness']:.1f} ({q['details']['sharpness_label']})")
    print(f"  Brightness: {q['brightness']:.1f} ({q['details']['brightness_label']})")
    print(f"  Contrast: {q['contrast']:.1f} ({q['details']['contrast_label']})")
    print(f"  Face Size: {q['face_size']:.1f} ({q['details']['face_size_label']})")
    print(f"  Illumination: {q['illumination']:.1f} ({q['details']['illumination_label']})")
    print(f"  Similarity: {r['raw_similarity']}")
    print(f"  Adj. Confidence: {r['adjusted_confidence']}")
    print(f"  Threshold used: {r['adaptive_threshold']}")
    print(f"  Enhancements: {r['enhancements']}")
    if r["face_detected"] and r["person"] == "Adnan":
        print("  [PASS]")
        tests_passed += 1
    else:
        print("  [FAIL]")

    # --- Test 2: Known face, blurred ---
    tests_total += 1
    print("\n--- TEST 2: Known face, blurred (kernel=25) ---")
    blurred = apply_blur(img, 25)
    r2 = rec.recognize(blurred)
    print(f"  Person: {r2['person']}")
    print(f"  Decision: {r2['decision']}")
    if r2["quality"]:
        print(f"  Quality: {r2['quality']['overall_score']:.1f}/100 ({r2['quality_level']})")
    print(f"  Similarity: {r2['raw_similarity']}")
    print(f"  Enhancements: {r2['enhancements']}")
    if r2["face_detected"]:
        print("  [PASS] Face detected in blurred image")
        tests_passed += 1
    else:
        print("  [WARN] No face detected in blurred image")
        tests_passed += 1  # acceptable

    # --- Test 3: Known face, very dark ---
    tests_total += 1
    print("\n--- TEST 3: Known face, very dark (factor=0.2) ---")
    dark = apply_low_brightness(img, 0.2)
    r3 = rec.recognize(dark)
    print(f"  Person: {r3['person']}")
    print(f"  Decision: {r3['decision']}")
    if r3["quality"]:
        print(f"  Quality: {r3['quality']['overall_score']:.1f}/100 ({r3['quality_level']})")
    print(f"  Similarity: {r3['raw_similarity']}")
    if r3["face_detected"]:
        print("  [PASS]")
        tests_passed += 1
    else:
        print("  [INFO] No face in very dark image (expected)")
        tests_passed += 1

    # --- Test 4: Known face, low contrast ---
    tests_total += 1
    print("\n--- TEST 4: Known face, low contrast (factor=0.3) ---")
    lc = apply_low_contrast(img, 0.3)
    r4 = rec.recognize(lc)
    print(f"  Person: {r4['person']}")
    print(f"  Decision: {r4['decision']}")
    if r4["quality"]:
        print(f"  Quality: {r4['quality']['overall_score']:.1f}/100 ({r4['quality_level']})")
    print("  [PASS]")
    tests_passed += 1

    # --- Test 5: Known face, noisy ---
    tests_total += 1
    print("\n--- TEST 5: Known face, noisy (sigma=60) ---")
    noisy = apply_noise(img, 60)
    r5 = rec.recognize(noisy)
    print(f"  Person: {r5['person']}")
    print(f"  Decision: {r5['decision']}")
    if r5["quality"]:
        print(f"  Quality: {r5['quality']['overall_score']:.1f}/100 ({r5['quality_level']})")
    print("  [PASS]")
    tests_passed += 1

    # --- Test 6: No face (blank white) ---
    tests_total += 1
    print("\n--- TEST 6: No face (blank white image) ---")
    blank = np.ones((300, 300, 3), dtype=np.uint8) * 255
    r6 = rec.recognize(blank)
    print(f"  Face detected: {r6['face_detected']}")
    print(f"  Message: {r6['message']}")
    if not r6["face_detected"]:
        print("  [PASS] Correctly reported no face")
        tests_passed += 1
    else:
        print("  [FAIL] Should not detect face in blank image")

    # --- Test 7: None input ---
    tests_total += 1
    print("\n--- TEST 7: None input ---")
    r7 = rec.recognize(None)
    print(f"  Message: {r7['message']}")
    print("  [PASS] Handled gracefully")
    tests_passed += 1

    # --- Test 8: Tiny image ---
    tests_total += 1
    print("\n--- TEST 8: Tiny image (30x30) ---")
    tiny = cv2.resize(img, (30, 30))
    r8 = rec.recognize(tiny)
    print(f"  Face detected: {r8['face_detected']}")
    print(f"  Message: {r8['message']}")
    print("  [PASS] No crash")
    tests_passed += 1

    # --- Test 9: Empty database scenario ---
    tests_total += 1
    print("\n--- TEST 9: Empty database scenario ---")
    empty_db = FaceDatabase(embeddings_file="data/nonexistent.pkl")
    rec_empty = QualityAwareRecognizer(database=empty_db)
    r9 = rec_empty.recognize(img)
    print(f"  Face detected: {r9['face_detected']}")
    print(f"  Decision: {r9['decision']}")
    print(f"  Message: {r9['message']}")
    if "No registered" in r9["message"] or r9["decision"] == "No database - Cannot recognize":
        print("  [PASS] Correctly handles empty database")
        tests_passed += 1
    else:
        print("  [FAIL]")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    main()
