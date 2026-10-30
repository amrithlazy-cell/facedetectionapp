import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import time
import streamlit.components.v1 as components

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FaceScope AI - Face Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #0f172a;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cascade Classifier Loader
# ---------------------------------------------------------
@st.cache_resource
def load_cascades():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascades_dir = os.path.join(base_dir, "cascades")

    def resolve(filename):
        local_p = os.path.join(cascades_dir, filename)
        if os.path.exists(local_p):
            return local_p
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cv_p = os.path.join(cv2.data.haarcascades, filename)
            if os.path.exists(cv_p):
                return cv_p
        return filename

    cascades = {
        "face_default": cv2.CascadeClassifier(resolve("haarcascade_frontalface_default.xml")),
        "face_alt2": cv2.CascadeClassifier(resolve("haarcascade_frontalface_alt2.xml")),
        "eye": cv2.CascadeClassifier(resolve("haarcascade_eye.xml")),
        "smile": cv2.CascadeClassifier(resolve("haarcascade_smile.xml")),
    }
    return cascades

cascades = load_cascades()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def process_face_detection(image_np, model_choice, scale_factor, min_neighbors, min_size,
                           detect_eyes, detect_smiles, blur_faces, box_color, box_thickness):
    # Prepare image
    h, w = image_np.shape[:2]
    result_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Select classifier
    face_clf = cascades["face_alt2"] if "Alt2" in model_choice else cascades["face_default"]

    start_time = time.time()
    faces = face_clf.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    proc_time = (time.time() - start_time) * 1000

    cropped_faces = []
    total_eyes = 0
    total_smiles = 0

    for idx, (fx, fy, fw, fh) in enumerate(faces):
        # Extract face crop
        crop = image_np[fy:fy+fh, fx:fx+fw]
        if crop.size > 0:
            cropped_faces.append(crop)

        if blur_faces:
            # Privacy mode: Blur the face ROI
            sub = result_bgr[fy:fy+fh, fx:fx+fw]
            ksize = max(15, (fw // 5) | 1)
            blurred = cv2.GaussianBlur(sub, (ksize, ksize), 30)
            result_bgr[fy:fy+fh, fx:fx+fw] = blurred
        else:
            # Draw bounding box
            cv2.rectangle(result_bgr, (fx, fy), (fx + fw, fy + fh), box_color, box_thickness)
            
            # Label
            label = f"Face #{idx+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_bgr, (fx, max(0, fy - 22)), (fx + label_size[0] + 12, max(0, fy)), box_color, cv2.FILLED)
            cv2.putText(result_bgr, label, (fx + 6, max(0, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            roi_bgr = result_bgr[fy:fy+fh, fx:fx+fw]

            # Eye Detection
            if detect_eyes and not cascades["eye"].empty():
                eyes = cascades["eye"].detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(max(15, fw // 8), max(15, fh // 8))
                )
                for (ex, ey, ew, eh) in eyes:
                    total_eyes += 1
                    eye_center = (ex + ew // 2, ey + eh // 2)
                    radius = int(round((ew + eh) * 0.25))
                    cv2.circle(roi_bgr, eye_center, radius, (255, 100, 0), max(1, box_thickness - 1))

            # Smile Detection
            if detect_smiles and not cascades["smile"].empty():
                smiles = cascades["smile"].detectMultiScale(
                    roi_gray,
                    scaleFactor=1.7,
                    minNeighbors=22,
                    minSize=(max(25, fw // 4), max(15, fh // 6))
                )
                for (sx, sy, sw, sh) in smiles:
                    total_smiles += 1
                    cv2.rectangle(roi_bgr, (sx, sy), (sx + sw, sy + sh), (0, 200, 50), 1)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb, len(faces), total_eyes, total_smiles, proc_time, cropped_faces

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    
    model_choice = st.selectbox(
        "Classifier Model",
        ["Frontal Face (Default - Balanced)", "Frontal Face (Alt2 - High Accuracy)"],
        index=0,
        help="Haar Cascade XML model used for face detection."
    )

    st.markdown("### Detection Parameters")
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.05,
        max_value=1.40,
        value=1.10,
        step=0.05,
        help="Specifies how much the image size is reduced at each image scale."
    )
    
    min_neighbors = st.slider(
        "Min Neighbors",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of neighboring candidate rectangles required to retain a detection."
    )
    
    min_face_size = st.slider(
        "Min Face Size (px)",
        min_value=20,
        max_value=120,
        value=30,
        step=5,
        help="Minimum dimensions of a detected face."
    )

    st.markdown("### Facial Feature Addons")
    detect_eyes = st.checkbox("Detect Eyes", value=True)
    detect_smiles = st.checkbox("Detect Smiles", value=False)

    st.markdown("### Appearance & Privacy")
    blur_faces = st.checkbox("Privacy Mode (Blur Faces)", value=False)
    box_color_hex = st.color_picker("Bounding Box Color", value="#06b6d4")
    box_color = hex_to_bgr(box_color_hex)
    box_thickness = st.slider("Box Thickness", min_value=1, max_value=5, value=2)

    st.markdown("---")
    st.caption("FaceScope AI v1.0 | Computer Vision Project")

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">FaceScope AI &mdash; Face Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Real-Time Face and Facial Feature Detection powered by Computer Vision.</div>', unsafe_allow_html=True)

# Tabs
tab_upload, tab_webcam_snap, tab_live_stream, tab_about = st.tabs([
    "📸 Image Upload",
    "📷 Camera Snapshot",
    "🎥 Live Real-Time Video",
    "ℹ️ About & Architecture"
])

# ---------------------------------------------------------
# TAB 1: Image Upload
# ---------------------------------------------------------
with tab_upload:
    col_ctrl, col_sample = st.columns([2, 1])
    with col_ctrl:
        uploaded_file = st.file_uploader(
            "Upload an image file (JPG, PNG, JPEG, WebP):",
            type=["jpg", "jpeg", "png", "webp"]
        )

    with col_sample:
        st.markdown("**Or pick a built-in sample:**")
        sample_choice = st.selectbox(
            "Sample Images",
            ["None", "Sample 1: Single Portrait", "Sample 2: Group of People"],
            index=0
        )

    # Resolve target image
    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
    elif sample_choice != "None":
        sample_filename = "sample_portrait.jpg" if "Single" in sample_choice else "sample_group.jpg"
        sample_path = os.path.join(os.path.dirname(__file__), "samples", sample_filename)
        if os.path.exists(sample_path):
            input_image = Image.open(sample_path).convert("RGB")

    if input_image is not None:
        img_np = np.array(input_image)

        # Run Face Detection
        with st.spinner("Processing image..."):
            processed_img, face_count, eyes_count, smiles_count, proc_time, crops = process_face_detection(
                img_np, model_choice, scale_factor, min_neighbors, min_face_size,
                detect_eyes, detect_smiles, blur_faces, box_color, box_thickness
            )

        # Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{face_count}</div>
                <div class="metric-label">Faces Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{eyes_count}</div>
                <div class="metric-label">Eyes Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{smiles_count}</div>
                <div class="metric-label">Smiles Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{proc_time:.1f} ms</div>
                <div class="metric-label">Latency</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side by Side comparison
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.subheader("Original Image")
            st.image(input_image, use_container_width=True)
        with col_proc:
            st.subheader("Detected Overlay")
            st.image(processed_img, use_container_width=True)

            # Download button for processed image
            buf = io.BytesIO()
            Image.fromarray(processed_img).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Processed Image",
                data=buf.getvalue(),
                file_name="facescope_detection.png",
                mime="image/png"
            )

        # Cropped Faces Gallery
        if len(crops) > 0:
            st.markdown("### 👤 Cropped Faces Gallery")
            crop_cols = st.columns(min(len(crops), 6))
            for i, crop in enumerate(crops):
                with crop_cols[i % len(crop_cols)]:
                    st.image(crop, caption=f"Face #{i+1} ({crop.shape[1]}x{crop.shape[0]})", use_container_width=True)
                    crop_buf = io.BytesIO()
                    Image.fromarray(crop).save(crop_buf, format="PNG")
                    st.download_button(
                        label=f"Save #{i+1}",
                        data=crop_buf.getvalue(),
                        file_name=f"face_{i+1}.png",
                        mime="image/png",
                        key=f"dl_crop_{i}"
                    )
    else:
        st.info("👆 Upload an image or select a sample image above to start face detection.")

# ---------------------------------------------------------
# TAB 2: Camera Snapshot
# ---------------------------------------------------------
with tab_webcam_snap:
    st.markdown("### Capture Photo from Webcam")
    st.write("Take a snapshot using your connected webcam or smartphone camera.")
    
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        snap_image = Image.open(camera_photo).convert("RGB")
        snap_np = np.array(snap_image)

        with st.spinner("Analyzing camera frame..."):
            proc_snap, snap_faces, snap_eyes, snap_smiles, snap_time, snap_crops = process_face_detection(
                snap_np, model_choice, scale_factor, min_neighbors, min_face_size,
                detect_eyes, detect_smiles, blur_faces, box_color, box_thickness
            )

        # Metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("Faces Detected", snap_faces)
        with sm2:
            st.metric("Eyes Detected", snap_eyes)
        with sm3:
            st.metric("Smiles Detected", snap_smiles)
        with sm4:
            st.metric("Inference Time", f"{snap_time:.1f} ms")

        # Result Display
        c1, c2 = st.columns(2)
        with c1:
            st.image(snap_image, caption="Captured Snapshot", use_container_width=True)
        with c2:
            st.image(proc_snap, caption="Detection Result", use_container_width=True)
            snap_buf = io.BytesIO()
            Image.fromarray(proc_snap).save(snap_buf, format="PNG")
            st.download_button("⬇️ Download Result", snap_buf.getvalue(), "webcam_face.png", "image/png")

# ---------------------------------------------------------
# TAB 3: Live Real-Time Video
# ---------------------------------------------------------
with tab_live_stream:
    st.markdown("### 🎥 Live Real-Time Browser Video Stream")
    st.write("Real-time 30 FPS face detection executing directly within your browser using client-side AI.")

    live_html = """
    <div style="background:#0f172a; border-radius:12px; padding:16px; border:1px solid #334155; text-align:center;">
        <div style="margin-bottom:12px; display:flex; justify-content:center; gap:10px;">
            <button id="btnStart" style="background:#06b6d4; color:#0f172a; font-weight:700; border:none; padding:8px 18px; border-radius:8px; cursor:pointer;">Start Camera</button>
            <button id="btnStop" style="background:#475569; color:#fff; font-weight:600; border:none; padding:8px 18px; border-radius:8px; cursor:pointer;">Stop Camera</button>
            <span id="liveBadge" style="background:#1e293b; color:#38bdf8; padding:8px 14px; border-radius:8px; font-size:12px; border:1px solid #334155;">Status: Ready</span>
        </div>
        <video id="liveVideo" playsinline style="display:none;"></video>
        <canvas id="liveCanvas" width="640" height="480" style="max-width:100%; border-radius:8px; background:#020617;"></canvas>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js" crossorigin="anonymous"></script>
    <script>
        const v = document.getElementById('liveVideo');
        const c = document.getElementById('liveCanvas');
        const ctx = c.getContext('2d');
        const badge = document.getElementById('liveBadge');
        let cam = null;

        const faceDet = new FaceDetection({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
        });
        faceDet.setOptions({ model: 'short', minDetectionConfidence: 0.55 });
        faceDet.onResults((res) => {
            ctx.save();
            ctx.clearRect(0, 0, c.width, c.height);
            ctx.drawImage(res.image, 0, 0, c.width, c.height);
            const dets = res.detections || [];
            badge.innerText = `Faces: ${dets.length} | Live`;
            dets.forEach(d => {
                const b = d.boundingBox;
                const x = b.xCenter * c.width - (b.width * c.width)/2;
                const y = b.yCenter * c.height - (b.height * c.height)/2;
                const w = b.width * c.width;
                const h = b.height * c.height;
                ctx.strokeStyle = '#06b6d4';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);
                if (d.landmarks) {
                    d.landmarks.forEach(lm => {
                        ctx.beginPath();
                        ctx.arc(lm.x * c.width, lm.y * c.height, 3, 0, 2*Math.PI);
                        ctx.fillStyle = '#f43f5e';
                        ctx.fill();
                    });
                }
            });
            ctx.restore();
        });

        document.getElementById('btnStart').onclick = async () => {
            try {
                badge.innerText = 'Connecting camera...';
                cam = new Camera(v, {
                    onFrame: async () => { await faceDet.send({ image: v }); },
                    width: 640, height: 480
                });
                await cam.start();
                badge.innerText = 'Streaming';
            } catch(e) {
                badge.innerText = 'Error: ' + e.message;
            }
        };

        document.getElementById('btnStop').onclick = () => {
            if (cam) { cam.stop(); cam = null; }
            badge.innerText = 'Stopped';
            ctx.clearRect(0, 0, c.width, c.height);
        };
    </script>
    """
    components.html(live_html, height=560)

# ---------------------------------------------------------
# TAB 4: About & Architecture
# ---------------------------------------------------------
with tab_about:
    st.markdown("""
    ### 🔬 Viola-Jones Face Detection Algorithm

    This application utilizes the **Viola-Jones Object Detection Framework**, an acclaimed computer vision algorithm proposed in 2001 by Paul Viola and Michael Jones.

    #### Key Pillars of the Architecture:
    1. **Haar-like Features:**
       - Digital image features used in object recognition.
       - Edge features, line features, and center-surround features that compute differences between pixel sums in adjacent rectangular regions.
    
    2. **Integral Image Representation:**
       - An intermediate image representation that allows any rectangular area sum to be evaluated in constant time $O(1)$, regardless of size.

    3. **AdaBoost Machine Learning:**
       - Selects a small number of critical features from hundreds of thousands of candidate features.
       - Constructs a strong classifier as a linear combination of weak classifiers.

    4. **Attentional Cascade:**
       - Multiple stages of classifiers arranged in a cascade hierarchy.
       - Sub-windows that fail early stages are instantly discarded, allowing the algorithm to focus computational power only on promising candidate face regions.

    ---
    ### 🚀 Deployment Options
    - **Streamlit Community Cloud:** Free instant hosting connected directly to your GitHub repository.
    - **GitHub Pages:** Static single-page web app using client-side AI (`index.html`).
    """)
