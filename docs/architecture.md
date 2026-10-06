# System Architecture
# Quality-Aware Adaptive Learning for Robust Face Recognition

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE / WEBCAM                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FACE DETECTION (OpenCV Haar Cascade)            │
│  • Detects face location(s) in the image                     │
│  • Returns bounding box (x, y, w, h)                         │
│  • Handles multiple faces (uses largest)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FACE CROPPING & PREPROCESSING                   │
│  • Crops face region with margin                             │
│  • Prepares image for quality assessment                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FACE QUALITY ASSESSMENT                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Sharpness   │  │  Brightness  │  │  Contrast    │       │
│  │  (Laplacian) │  │  (Mean)      │  │  (Std Dev)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐                          │
│  │  Face Size   │  │ Illumination │                          │
│  │  (pixels)    │  │ (L/R ratio)  │                          │
│  └──────────────┘  └──────────────┘                          │
│                                                              │
│  → Weighted Average → Quality Score (0-100)                  │
│  → Quality Level: High (80+) / Medium (60-79) / Low (<60)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         QUALITY-AWARE ADAPTIVE PREPROCESSING                 │
│                                                              │
│  HIGH Quality:    Standard normalization only                │
│  MEDIUM Quality:  Brightness/Contrast enhancement            │
│  LOW Quality:     Aggressive enhancement + CLAHE +           │
│                   Sharpening + Denoising                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           FACE EMBEDDING EXTRACTION                          │
│  • OpenCV SFace model (128-dimensional vector)               │
│  • Captures unique facial features as numbers                │
│  • Fallback: Histogram-based features                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         SIMILARITY COMPARISON WITH DATABASE                  │
│  • Cosine similarity between embeddings                      │
│  • Compare with all registered persons                       │
│  • Find best match and similarity score                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│       QUALITY-AWARE CONFIDENCE ADJUSTMENT                    │
│                                                              │
│  HIGH Quality:    threshold = 0.45, confidence × 1.00        │
│  MEDIUM Quality:  threshold = 0.52, confidence × 0.85        │
│  LOW Quality:     threshold = 0.60, confidence × 0.65        │
│                                                              │
│  Key Insight: Low-quality images require HIGHER similarity   │
│  to be accepted, reducing false positive recognitions.       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL ADAPTIVE DECISION                         │
│                                                              │
│  IF similarity >= adaptive_threshold:                        │
│     → RECOGNIZED (Person Name + Confidence)                  │
│  ELSE:                                                       │
│     → UNKNOWN PERSON                                         │
│                                                              │
│  + Quality warning for low-quality recognition               │
└─────────────────────────────────────────────────────────────┘
```

## Quality Assessment Metrics

| Metric       | Method                  | Weight | What It Measures                    |
|-------------|-------------------------|--------|-------------------------------------|
| Sharpness   | Laplacian Variance      | 30%    | Image focus / blur level            |
| Brightness  | Mean pixel intensity    | 20%    | Too dark or too bright              |
| Contrast    | Standard deviation      | 20%    | Dynamic range of image              |
| Face Size   | Pixel dimensions        | 15%    | Face resolution for recognition     |
| Illumination| Left/right brightness   | 15%    | Lighting uniformity across face     |

## Adaptive Threshold Logic

| Quality Level | Threshold | Confidence Multiplier | Reasoning                             |
|--------------|-----------|----------------------|---------------------------------------|
| High (80+)   | 0.45      | 1.00 (100%)          | Trust the embedding fully             |
| Medium (60-79)| 0.52     | 0.85 (85%)           | Some degradation, moderate caution    |
| Low (<60)    | 0.60      | 0.65 (65%)           | Unreliable features, high caution     |

## Technology Stack

- **Python 3.10+**
- **OpenCV** — Face detection (Haar Cascade) + Face recognition (SFace)
- **NumPy** — Numerical computations
- **Streamlit** — Web interface
- **Pillow** — Image handling
