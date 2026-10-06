# Quality-Aware Adaptive Learning for Robust Face Recognition

> AI-Based Face Recognition System with Image Quality Assessment and Adaptive Confidence

---

## 📌 Problem Statement

Traditional face recognition systems treat all input images equally, regardless of their quality. This can lead to unreliable recognition when images are blurry, poorly lit, or low resolution. Our system addresses this by **assessing image quality before recognition** and **adapting the decision threshold** based on quality — making recognition more robust and reliable.

---

## 🎯 Objectives

1. **Detect human faces** from images using computer vision
2. **Recognize registered individuals** using face embeddings
3. **Assess image quality** using multiple metrics (sharpness, brightness, contrast, etc.)
4. **Adapt recognition decisions** based on quality — the core contribution
5. **Provide a user-friendly web interface** for testing and demonstration

---

## 💡 How It Works

### The Key Idea

> Not every face image has the same recognition reliability. A blurry or dark photo should not be trusted as much as a clear, well-lit one.

Our system implements **Quality-Aware Adaptive Recognition**:

1. **Assess Quality First**: Before trying to recognize a face, the system evaluates the image quality using 5 metrics
2. **Preprocess Adaptively**: Low-quality images receive more enhancement (contrast, brightness, sharpening)
3. **Adjust Thresholds**: The recognition threshold changes based on quality level:
   - **High quality**: Lower threshold (more lenient) — trust the embedding
   - **Medium quality**: Moderate threshold — some caution
   - **Low quality**: Higher threshold (stricter) — reduce trust in unreliable features

### Pipeline

```
Input Image → Face Detection → Face Cropping → Quality Assessment →
Quality Score → Adaptive Preprocessing → Face Embedding →
Similarity Comparison → Quality-Aware Confidence → Final Decision
```

### Quality Metrics

| Metric | Method | What It Measures |
|--------|--------|-----------------|
| Sharpness | Laplacian variance | Focus / blur level |
| Brightness | Mean pixel intensity | Too dark or bright |
| Contrast | Standard deviation | Dynamic range |
| Face Size | Pixel dimensions | Resolution |
| Illumination | L/R brightness ratio | Lighting uniformity |

### Adaptive Thresholds

| Quality Level | Score Range | Recognition Threshold | Confidence |
|--------------|-------------|----------------------|------------|
| 🟢 High | 80-100 | 0.45 (lenient) | 100% |
| 🟡 Medium | 60-79 | 0.52 (moderate) | 85% |
| 🔴 Low | 0-59 | 0.60 (strict) | 65% |

---

## 🏗️ Architecture

```
quality-aware-face-recognition/
│
├── app.py                          # Main Streamlit web application
├── train.py                        # Original model training script
├── requirements.txt                # Python dependencies
├── setup.bat                       # One-click setup (Windows)
├── run_demo.bat                    # One-click launch (Windows)
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── face_detector.py            # Face detection (OpenCV Haar Cascade)
│   ├── face_recognizer.py          # Face embeddings (OpenCV SFace)
│   ├── quality_assessment.py       # Quality scoring module
│   ├── quality_aware_recognizer.py # Core: Quality-aware recognition pipeline
│   ├── preprocessing.py            # Adaptive image enhancement
│   ├── database.py                 # Face embeddings database
│   ├── utils.py                    # Helper functions
│   ├── face_model.py               # Original TF/Keras model (reference)
│   └── image_processing.py         # Original image processing (reference)
│
├── scripts/
│   └── build_database.py           # Build face embeddings database
│
├── dataset/
│   └── registered/                 # Put face images here
│       ├── PersonName1/
│       │   ├── image1.jpg
│       │   └── image2.jpg
│       └── PersonName2/
│           └── image1.jpg
│
├── data/
│   └── embeddings.pkl              # Generated face database
│
├── models/                         # Model files (auto-downloaded)
│   ├── face_model.h5               # Original trained model
│   └── face_recognition_sface_2021dec.onnx  # SFace model (auto-download)
│
├── docs/
│   └── architecture.md             # System architecture documentation
│
└── results/                        # Results and outputs
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Programming language |
| **OpenCV** | Face detection, embedding extraction (SFace), image processing |
| **NumPy** | Numerical computations |
| **Streamlit** | Web interface |
| **Pillow** | Image handling |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Windows 10/11 (tested), also works on macOS/Linux
- No GPU required

### Quick Setup (Windows)

```batch
# Option 1: Double-click setup.bat

# Option 2: Manual setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Setup on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📸 Dataset Setup

### Step 1: Add Face Images

Create folders inside `dataset/registered/` — one folder per person:

```
dataset/registered/
    Fuzail/
        photo1.jpg
        photo2.jpg
        photo3.jpg
    Adnan/
        photo1.jpg
    Saif/
        photo1.jpg
        photo2.jpg
```

**Tips:**
- Use **1-5 clear face images** per person
- Good lighting, front-facing preferred
- Different angles/expressions help recognition
- JPEG, PNG, BMP formats supported

### Step 2: Build Face Database

```bash
python scripts/build_database.py
```

Expected output:
```
============================================================
  Face Database Builder
  Quality-Aware Face Recognition System
============================================================

[INFO] Initializing face recognition components...
[INFO] SFace recognizer loaded successfully

Found 3 identities
----------------------------------------

Fuzail -> 3 images
  [OK] photo1.jpg - Quality: 87/100 (High)
  [OK] photo2.jpg - Quality: 74/100 (Medium)
  [OK] photo3.jpg - Quality: 91/100 (High)

Adnan -> 1 images
  [OK] photo1.jpg - Quality: 85/100 (High)

Saif -> 2 images
  [OK] photo1.jpg - Quality: 82/100 (High)
  [OK] photo2.jpg - Quality: 78/100 (Medium)

============================================================

✅ Database created successfully!
   Total identities: 3
   Total images processed: 6
   Total embeddings stored: 6
```

---

## ▶️ How to Run

```bash
# Activate virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Start the application
streamlit run app.py
```

Or simply double-click **`run_demo.bat`** on Windows.

The web interface will open at `http://localhost:8501`

---

## 🎮 Demo Instructions

### Demo Flow (5-minute presentation):

1. **Open the app** → Show the Dashboard page
2. **Navigate to "Recognize Face"**
3. **Upload a registered person's good photo** → Show:
   - Face detected with bounding box
   - Quality score (should be high, ~80-90)
   - Recognition: Person recognized ✅
   - High confidence score
4. **Navigate to "Quality Simulator"**
5. **Upload the same photo and apply blur** → Show:
   - Quality score drops significantly
   - Recognition may fail or show lower confidence
   - System shows "Low Quality — Recognition Less Reliable"
   - Adaptive preprocessing is applied
6. **Upload an unknown person's photo** → Show:
   - Unknown Person ❌
   - No match in database
7. **Explain the key concept**: "The system adapts its recognition threshold based on image quality. High-quality images are trusted more, while low-quality images require stricter similarity to prevent false recognition."

---

## 📊 Expected Output

### High Quality Recognition
```
Person: Fuzail
Recognition Confidence: 91%
Face Quality: 87/100
Quality Level: HIGH
Decision: ✅ RECOGNIZED
```

### Low Quality Recognition
```
Person: Fuzail
Raw Similarity: 72%
Face Quality: 38/100
Quality Level: LOW
Adaptive Decision: LOW QUALITY — Recognition Less Reliable
```

### Unknown Person
```
Person: Unknown
Confidence: 31%
Quality: 78/100
Decision: ❌ UNKNOWN PERSON
```

---

## ⚠️ Limitations

1. **Face Detection**: Uses Haar Cascade which may miss faces at extreme angles or in very poor lighting
2. **Single Face Focus**: Processes only the largest face when multiple faces are detected
3. **Quality Assessment**: Uses practical metrics (not a deep-learning quality model) — sufficient for demonstration
4. **Recognition Accuracy**: Depends on the quality and variety of registered images
5. **Model**: Uses OpenCV's SFace model — production systems would use more advanced models

---

## 🔮 Future Scope

1. Deep learning-based face quality assessment (e.g., SER-FIQ)
2. More advanced face recognition models (ArcFace, CosFace)
3. Real-time video stream processing
4. Multi-face simultaneous recognition
5. Face anti-spoofing (liveness detection)
6. Cloud deployment for scalability
7. Integration with attendance management systems

---

## 👨‍💻 Team

- **Adnan Ansari**
- **Saif**
- **Fuzail**

---

## 📜 License

This project is developed for academic and educational purposes.