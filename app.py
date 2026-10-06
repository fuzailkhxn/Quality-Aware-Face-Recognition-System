"""
Quality-Aware Adaptive Learning for Robust Face Recognition
Main Streamlit Application

This is the main entry point for the web interface.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.face_detector import FaceDetector
from src.quality_assessment import FaceQualityAssessor
from src.preprocessing import FacePreprocessor
from src.face_recognizer import FaceEmbedder
from src.database import FaceDatabase
from src.quality_aware_recognizer import QualityAwareRecognizer
from src.utils import (
    load_image_from_upload, bgr_to_rgb,
    apply_blur, apply_low_brightness, apply_low_contrast, apply_noise
)

# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Quality-Aware Face Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin-bottom: 0.3rem; }
    .main-header p { color: #a0c4ff; font-size: 0.95rem; margin: 0; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #0f3460;
    }
    .quality-high { border-left-color: #28a745 !important; }
    .quality-medium { border-left-color: #ffc107 !important; }
    .quality-low { border-left-color: #dc3545 !important; }
    .result-recognized {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-unknown {
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-noface {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Components (cached) ───────────────────────────────
@st.cache_resource
def get_recognizer():
    """Initialize the quality-aware recognizer (cached)."""
    db = FaceDatabase()
    recognizer = QualityAwareRecognizer(database=db)
    return recognizer


@st.cache_resource
def get_database():
    """Get the face database (cached)."""
    return FaceDatabase()


def reload_database():
    """Reload database - clear cache and reinitialize."""
    st.cache_resource.clear()


# ─── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Quality-Aware Adaptive Learning for Robust Face Recognition</h1>
    <p>AI-Based Face Recognition with Image Quality Assessment and Adaptive Confidence</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "📸 Recognize Face", "🧪 Quality Simulator", "👤 Register Person", "📊 Database Info"],
        index=1
    )
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    **Project:** Quality-Aware Adaptive Learning for Robust Face Recognition
    
    **Key Concept:** Not every face image has the same recognition reliability.
    The system first estimates face quality and then adapts the recognition
    decision based on the quality of the input.
    
    **Team:** Adnan, Saif, Fuzail
    """)
    
    st.divider()
    
    # Database status
    db = get_database()
    if db.is_empty():
        st.warning("⚠️ No faces registered. Add images to `dataset/registered/` and run `python scripts/build_database.py`")
    else:
        stats = db.get_stats()
        st.success(f"✅ {len(stats)} identities registered")
        for name, count in stats.items():
            st.text(f"  • {name}: {count} embeddings")


# ─── Dashboard Page ────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    db = get_database()
    stats = db.get_stats()
    
    with col1:
        st.metric("Registered Identities", len(stats))
    with col2:
        total_emb = sum(stats.values()) if stats else 0
        st.metric("Total Embeddings", total_emb)
    with col3:
        st.metric("Quality Levels", "3 (High/Med/Low)")
    with col4:
        st.metric("Adaptive Thresholds", "Yes ✓")
    
    st.divider()
    
    st.subheader("📋 System Architecture")
    st.markdown("""
    ```
    INPUT IMAGE
         ↓
    ┌─────────────────┐
    │  Face Detection  │ ← OpenCV Haar Cascade
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Face Cropping   │ ← With margin padding
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Quality Assessment│ ← Sharpness, Brightness, Contrast, Size, Illumination
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │Quality Score 0-100│ ← Weighted combination
    └────────┬────────┘
             ↓
    ┌─────────────────────┐
    │ Adaptive Preprocessing│ ← Enhancement based on quality level
    └────────┬────────────┘
             ↓
    ┌─────────────────┐
    │ Face Embedding   │ ← OpenCV SFace (128-d vector)
    └────────┬────────┘
             ↓
    ┌─────────────────────────┐
    │ Similarity Comparison    │ ← Cosine similarity with database
    └────────┬────────────────┘
             ↓
    ┌──────────────────────────┐
    │ Quality-Aware Confidence  │ ← Adjusted threshold per quality level
    └────────┬─────────────────┘
             ↓
    ┌─────────────────┐
    │ Final Decision   │ ← RECOGNIZED / UNKNOWN / UNRELIABLE
    └─────────────────┘
    ```
    """)
    
    st.divider()
    
    st.subheader("🎯 Quality-Aware Adaptive Logic")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🟢 High Quality (80-100)
        - Normal recognition threshold
        - Full confidence in result
        - Standard processing
        - Threshold: **0.45**
        """)
    with col2:
        st.markdown("""
        #### 🟡 Medium Quality (60-79)
        - Moderate enhancement applied
        - Slightly stricter threshold
        - Reduced confidence multiplier
        - Threshold: **0.52**
        """)
    with col3:
        st.markdown("""
        #### 🔴 Low Quality (0-59)
        - Aggressive enhancement
        - Much stricter threshold
        - Significantly reduced trust
        - Threshold: **0.60**
        """)


# ─── Recognize Face Page ──────────────────────────────────────────
elif page == "📸 Recognize Face":
    st.header("📸 Face Recognition")
    
    upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📷 Camera"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear photo of a person's face"
        )
        
        if uploaded_file is not None:
            # Load image
            image = load_image_from_upload(uploaded_file)
            
            if image is None:
                st.error("❌ Could not load image. Please upload a valid image file.")
            else:
                # Run recognition
                recognizer = get_recognizer()
                result = recognizer.recognize(image)
                
                # Display results
                col_img, col_result = st.columns([1, 1])
                
                with col_img:
                    st.subheader("📷 Input Image")
                    if result["annotated_image"] is not None:
                        st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)
                    else:
                        st.image(bgr_to_rgb(image), use_container_width=True)
                    
                    # Show cropped face
                    if result["face_crop"] is not None:
                        st.subheader("✂️ Detected Face")
                        st.image(bgr_to_rgb(result["face_crop"]), width=200)
                
                with col_result:
                    st.subheader("📊 Recognition Result")
                    
                    if not result["face_detected"]:
                        st.markdown(f"""
                        <div class="result-noface">
                            <h3>⚠️ No Face Detected</h3>
                            <p>{result['message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif result["decision"] == "RECOGNIZED":
                        st.markdown(f"""
                        <div class="result-recognized">
                            <h2>✅ RECOGNIZED</h2>
                            <h3>Person: {result['person']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif result["decision"] == "UNKNOWN PERSON":
                        st.markdown(f"""
                        <div class="result-unknown">
                            <h2>❌ UNKNOWN PERSON</h2>
                            <p>This person is not registered in the database.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        st.info(f"ℹ️ {result['decision']}")
                        st.write(result['message'])
                    
                    # Metrics
                    if result["face_detected"]:
                        st.divider()
                        
                        # Confidence and Quality
                        m1, m2 = st.columns(2)
                        with m1:
                            conf_pct = result['adjusted_confidence'] * 100
                            st.metric("Recognition Confidence", f"{conf_pct:.1f}%")
                            st.progress(min(1.0, result['adjusted_confidence']))
                        with m2:
                            q_score = result['quality']['overall_score'] if result['quality'] else 0
                            st.metric("Face Quality", f"{q_score:.0f}/100")
                            st.progress(min(1.0, q_score / 100))
                        
                        st.divider()
                        
                        # Quality Level
                        ql = result['quality_level']
                        if ql == "High":
                            st.success(f"🟢 Quality Level: **{ql}** — Full confidence recognition")
                        elif ql == "Medium":
                            st.warning(f"🟡 Quality Level: **{ql}** — Moderate confidence")
                        else:
                            st.error(f"🔴 Quality Level: **{ql}** — Recognition less reliable")
                        
                        # Decision details
                        if result["decision_details"]:
                            st.info(f"📋 {result['decision_details']}")
                        
                        # Enhancements applied
                        if result["enhancements"]:
                            st.write("**🔧 Preprocessing Applied:**")
                            for enh in result["enhancements"]:
                                st.write(f"  • {enh}")
                        
                        st.divider()
                        
                        # Quality Breakdown
                        st.subheader("📈 Quality Analysis")
                        if result["quality"]:
                            q = result["quality"]
                            
                            metrics = [
                                ("Sharpness", q["sharpness"], q["details"]["sharpness_label"]),
                                ("Brightness", q["brightness"], q["details"]["brightness_label"]),
                                ("Contrast", q["contrast"], q["details"]["contrast_label"]),
                                ("Face Size", q["face_size"], q["details"]["face_size_label"]),
                                ("Illumination", q["illumination"], q["details"]["illumination_label"]),
                            ]
                            
                            for metric_name, score, label in metrics:
                                col_a, col_b, col_c = st.columns([2, 3, 1])
                                with col_a:
                                    st.write(f"**{metric_name}**")
                                with col_b:
                                    st.progress(min(1.0, score / 100))
                                with col_c:
                                    if label == "Good":
                                        st.write(f"🟢 {score:.0f}")
                                    elif label == "Fair":
                                        st.write(f"🟡 {score:.0f}")
                                    else:
                                        st.write(f"🔴 {score:.0f}")
                        
                        # All similarities
                        if result["all_similarities"]:
                            st.divider()
                            st.subheader("🔍 Similarity Scores")
                            for name, sim in sorted(result["all_similarities"].items(),
                                                     key=lambda x: x[1], reverse=True):
                                col_a, col_b = st.columns([1, 3])
                                with col_a:
                                    st.write(f"**{name}**")
                                with col_b:
                                    st.progress(min(1.0, sim))
                                    st.caption(f"{sim * 100:.1f}%")
                        
                        # Adaptive threshold info
                        st.divider()
                        st.subheader("⚙️ Adaptive Threshold Details")
                        st.write(f"- **Quality Level:** {result['quality_level']}")
                        st.write(f"- **Adaptive Threshold:** {result['adaptive_threshold']:.2f}")
                        st.write(f"- **Raw Similarity:** {result['raw_similarity']:.4f}")
                        st.write(f"- **Adjusted Confidence:** {result['adjusted_confidence']:.4f}")
    
    with camera_tab:
        st.info("📷 Use your device's camera to take a photo, then upload it in the Upload tab.")
        camera_input = st.camera_input("Take a photo")
        
        if camera_input is not None:
            image = load_image_from_upload(camera_input)
            if image is not None:
                recognizer = get_recognizer()
                result = recognizer.recognize(image)
                
                col_img, col_result = st.columns([1, 1])
                
                with col_img:
                    if result["annotated_image"] is not None:
                        st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)
                
                with col_result:
                    if result["face_detected"]:
                        if result["decision"] == "RECOGNIZED":
                            st.success(f"✅ **{result['person']}** recognized!")
                        else:
                            st.error("❌ Unknown person")
                        
                        st.metric("Confidence", f"{result['adjusted_confidence']*100:.1f}%")
                        st.metric("Quality", f"{result['quality']['overall_score']:.0f}/100")
                    else:
                        st.warning(result["message"])


# ─── Quality Simulator Page ──────────────────────────────────────
elif page == "🧪 Quality Simulator":
    st.header("🧪 Image Quality Simulation")
    st.markdown("""
    This tool demonstrates how image quality affects recognition.
    Upload an image, apply degradations, and compare the results.
    """)
    
    sim_file = st.file_uploader(
        "Upload a face image to simulate quality changes",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="sim_upload"
    )
    
    if sim_file is not None:
        original = load_image_from_upload(sim_file)
        
        if original is None:
            st.error("Could not load image.")
        else:
            st.subheader("🎛️ Quality Degradation Controls")
            
            col_ctrl1, col_ctrl2 = st.columns(2)
            
            with col_ctrl1:
                blur_amount = st.slider("🌫️ Blur Amount", 0, 30, 0, step=2,
                                        help="Simulate motion blur or focus issues")
                brightness_factor = st.slider("🔆 Brightness", 0.1, 2.0, 1.0, step=0.1,
                                              help="Lower = darker image")
            
            with col_ctrl2:
                contrast_factor = st.slider("🎨 Contrast", 0.1, 2.0, 1.0, step=0.1,
                                            help="Lower = washed out")
                noise_amount = st.slider("📡 Noise", 0, 80, 0, step=5,
                                         help="Add random noise")
            
            # Apply degradations
            degraded = original.copy()
            applied_effects = []
            
            if blur_amount > 0:
                k = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
                degraded = apply_blur(degraded, k)
                applied_effects.append(f"Blur (kernel={k})")
            
            if brightness_factor != 1.0:
                degraded = cv2.convertScaleAbs(degraded, alpha=brightness_factor, beta=0)
                applied_effects.append(f"Brightness (×{brightness_factor:.1f})")
            
            if contrast_factor != 1.0:
                mean = np.mean(degraded)
                degraded = np.clip(
                    (degraded.astype(np.float32) - mean) * contrast_factor + mean,
                    0, 255
                ).astype(np.uint8)
                applied_effects.append(f"Contrast (×{contrast_factor:.1f})")
            
            if noise_amount > 0:
                degraded = apply_noise(degraded, noise_amount)
                applied_effects.append(f"Noise (σ={noise_amount})")
            
            # Run recognition on both
            recognizer = get_recognizer()
            result_original = recognizer.recognize(original)
            result_degraded = recognizer.recognize(degraded)
            
            # Display comparison
            st.divider()
            st.subheader("📊 Comparison: Original vs Degraded")
            
            col_orig, col_deg = st.columns(2)
            
            with col_orig:
                st.markdown("### ✅ Original")
                if result_original["annotated_image"] is not None:
                    st.image(bgr_to_rgb(result_original["annotated_image"]),
                             use_container_width=True)
                else:
                    st.image(bgr_to_rgb(original), use_container_width=True)
                
                if result_original["face_detected"]:
                    q = result_original["quality"]
                    st.metric("Quality Score", f"{q['overall_score']:.0f}/100")
                    st.metric("Quality Level", q['quality_level'])
                    st.metric("Person", result_original['person'])
                    st.metric("Confidence",
                              f"{result_original['adjusted_confidence']*100:.1f}%")
                    st.metric("Decision", result_original['decision'])
                else:
                    st.warning("No face detected")
            
            with col_deg:
                st.markdown("### 🔄 Degraded")
                if applied_effects:
                    st.caption(f"Effects: {', '.join(applied_effects)}")
                
                if result_degraded["annotated_image"] is not None:
                    st.image(bgr_to_rgb(result_degraded["annotated_image"]),
                             use_container_width=True)
                else:
                    st.image(bgr_to_rgb(degraded), use_container_width=True)
                
                if result_degraded["face_detected"]:
                    q = result_degraded["quality"]
                    
                    # Show quality change
                    orig_q = result_original.get("quality", {}).get("overall_score", 0)
                    delta_q = q['overall_score'] - orig_q
                    st.metric("Quality Score", f"{q['overall_score']:.0f}/100",
                              delta=f"{delta_q:.0f}")
                    st.metric("Quality Level", q['quality_level'])
                    st.metric("Person", result_degraded['person'])
                    
                    orig_conf = result_original.get("adjusted_confidence", 0) * 100
                    deg_conf = result_degraded['adjusted_confidence'] * 100
                    delta_conf = deg_conf - orig_conf
                    st.metric("Confidence", f"{deg_conf:.1f}%",
                              delta=f"{delta_conf:.1f}%")
                    st.metric("Decision", result_degraded['decision'])
                    
                    # Show enhancements
                    if result_degraded["enhancements"]:
                        st.write("**Adaptive Preprocessing Applied:**")
                        for enh in result_degraded["enhancements"]:
                            st.write(f"  • {enh}")
                else:
                    st.error("No face detected after degradation")
            
            # Quick presets
            st.divider()
            st.subheader("⚡ Quick Presets")
            
            p1, p2, p3, p4 = st.columns(4)
            
            with p1:
                if st.button("🌫️ Heavy Blur", use_container_width=True):
                    st.query_params["blur"] = "25"
                    st.rerun()
            with p2:
                if st.button("🌑 Very Dark", use_container_width=True):
                    st.query_params["dark"] = "1"
                    st.rerun()
            with p3:
                if st.button("🎨 Low Contrast", use_container_width=True):
                    st.query_params["contrast"] = "1"
                    st.rerun()
            with p4:
                if st.button("📡 Heavy Noise", use_container_width=True):
                    st.query_params["noise"] = "1"
                    st.rerun()


# ─── Register Person Page ─────────────────────────────────────────
elif page == "👤 Register Person":
    st.header("👤 Register New Person")
    
    registered_dir = os.path.join(PROJECT_ROOT, "dataset", "registered")
    
    st.markdown("""
    ### How to Register:
    
    **Option 1: Upload through this interface**
    1. Enter the person's name below
    2. Upload 1-3 clear face images
    3. Click "Register"
    4. Click "Rebuild Database"
    
    **Option 2: Manual folder setup**
    1. Create a folder: `dataset/registered/PersonName/`
    2. Put 1-3 face images in it
    3. Run: `python scripts/build_database.py`
    """)
    
    st.divider()
    
    person_name = st.text_input("Person's Name", placeholder="e.g., Fuzail")
    
    reg_files = st.file_uploader(
        "Upload face images (1-5 clear photos)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="reg_upload"
    )
    
    if st.button("📥 Register Person", type="primary", disabled=not person_name):
        if not person_name.strip():
            st.error("Please enter a name.")
        elif not reg_files:
            st.error("Please upload at least one image.")
        else:
            # Save images
            person_dir = os.path.join(registered_dir, person_name.strip())
            os.makedirs(person_dir, exist_ok=True)
            
            saved = 0
            for i, f in enumerate(reg_files):
                try:
                    img_data = f.read()
                    img_path = os.path.join(person_dir, f"image_{i+1}.jpg")
                    with open(img_path, 'wb') as out:
                        out.write(img_data)
                    saved += 1
                    f.seek(0)
                except Exception as e:
                    st.error(f"Error saving {f.name}: {e}")
            
            if saved > 0:
                st.success(f"✅ Saved {saved} images for {person_name.strip()} to `{person_dir}`")
                st.info("⚠️ Now click **Rebuild Database** below to update face embeddings.")
    
    st.divider()
    
    if st.button("🔄 Rebuild Database", type="secondary"):
        with st.spinner("Building face database..."):
            try:
                # Run the build_database script inline
                from scripts.build_database import build_database
                build_database(registered_dir)
                reload_database()
                st.success("✅ Database rebuilt successfully! Refresh the page to see updated results.")
                st.rerun()
            except Exception as e:
                st.error(f"Error building database: {e}")
                st.info("Try running manually: `python scripts/build_database.py`")
    
    # Show currently registered
    st.divider()
    st.subheader("📂 Currently Registered Persons")
    
    if os.path.exists(registered_dir):
        persons = [d for d in os.listdir(registered_dir)
                   if os.path.isdir(os.path.join(registered_dir, d))]
        
        if persons:
            for person in sorted(persons):
                person_dir = os.path.join(registered_dir, person)
                images = [f for f in os.listdir(person_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
                
                with st.expander(f"👤 {person} ({len(images)} images)"):
                    if images:
                        cols = st.columns(min(len(images), 4))
                        for i, img_name in enumerate(images[:4]):
                            img_path = os.path.join(person_dir, img_name)
                            with cols[i % len(cols)]:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    st.image(bgr_to_rgb(img), caption=img_name, width=150)
                    else:
                        st.write("No images found.")
        else:
            st.info("No persons registered yet. Add images above or create folders in `dataset/registered/`.")
    else:
        st.info(f"Directory `{registered_dir}` does not exist yet. It will be created when you register someone.")


# ─── Database Info Page ────────────────────────────────────────────
elif page == "📊 Database Info":
    st.header("📊 Database Information")
    
    db = get_database()
    
    if db.is_empty():
        st.warning("📭 Database is empty. No faces registered.")
        st.markdown("""
        ### Quick Setup:
        1. Create folder: `dataset/registered/YourName/`
        2. Add 1-3 face images
        3. Run: `python scripts/build_database.py`
        4. Refresh this page
        """)
    else:
        stats = db.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Identities", len(stats))
        with col2:
            st.metric("Total Embeddings", sum(stats.values()))
        
        st.divider()
        
        st.subheader("📋 Registered Identities")
        for name, count in sorted(stats.items()):
            st.write(f"**{name}**: {count} face embedding(s)")
        
        st.divider()
        
        if st.button("🗑️ Clear Database"):
            db.clear()
            reload_database()
            st.success("Database cleared.")
            st.rerun()


# ─── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    Quality-Aware Adaptive Learning for Robust Face Recognition | 
    College Project | Team: Adnan, Saif, Fuzail
</div>
""", unsafe_allow_html=True)
