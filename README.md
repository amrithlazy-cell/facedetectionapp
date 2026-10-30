# FaceScope AI &mdash; Real-Time Face Detection Application 👤⚡

A computer vision application for real-time face detection, facial landmark tracking, and privacy filtering. Built with **OpenCV**, **Streamlit**, and **MediaPipe Face Detection**.

---

## 🔗 Live Deployments

- **Streamlit Web Application:** [https://face-detection-app.streamlit.app](https://share.streamlit.io) *(Deployable via Streamlit Community Cloud)*
- **GitHub Pages Live Demo:** `https://amrithlazy-cell.github.io/facedetectionapp/` *(Instant client-side browser deployment)*

---

## ✨ Features

- **Multi-Mode Input:**
  - 📸 **Image Upload:** Upload JPG, PNG, or WebP images with instant face, eye, and smile detection.
  - 📷 **Webcam Snapshot:** Capture photos directly from your browser/mobile device using `st.camera_input`.
  - 🎥 **Real-Time Video Stream:** High-speed 30 FPS client-side video face detection with zero server lag.
  - 💻 **Desktop Native Script:** Standalone OpenCV live webcam detector (`detect_webcam.py`).
- **Precision Computer Vision:**
  - Haar Feature-based Cascade Classifiers (Frontal Face Default & Frontal Face Alt2).
  - Multi-scale detection with configurable scale factor, minimum neighbors, and minimum face size.
  - Eye and smile feature detection within candidate facial regions of interest (ROI).
- **Face Gallery & Cropping:**
  - Automatically isolates and extracts individual detected faces.
  - Grid preview with face dimensions and 1-click individual face downloads.
- **Privacy & Customization:**
  - **Privacy Mode (Gaussian Face Blur):** Automatically censors/blurs all detected faces.
  - Interactive bounding box color picker and stroke thickness slider.
  - Real-time latency (ms) and detection metrics.

---

## 🔬 Algorithm & Theory

The application leverages the **Viola-Jones Object Detection Framework** (2001), combining:
1. **Haar-like Features:** Pixel difference calculations across adjacent rectangular regions to detect facial contours, eye sockets, and bridge of the nose.
2. **Integral Images:** Enables constant-time $O(1)$ computation of rectangular sums, irrespective of window dimensions.
3. **AdaBoost Algorithm:** Selects the most distinctive features and constructs a weighted ensemble of weak classifiers.
4. **Attentional Cascade:** Discards non-face background windows rapidly in early stages, reserving heavy computation only for viable face candidates.

---

## 📁 Repository Structure

```text
face-detection-app/
├── app.py                # Main Streamlit web application
├── detect_webcam.py      # Standalone desktop OpenCV live webcam script
├── index.html            # Static client-side web application for GitHub Pages
├── requirements.txt      # Python dependencies
├── packages.txt          # System libraries for Linux cloud deployment
├── .gitignore            # Git ignore file
├── cascades/             # Pre-trained Haar Cascade XML classifiers
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_frontalface_alt2.xml
│   ├── haarcascade_eye.xml
│   └── haarcascade_smile.xml
├── samples/              # Test sample images for instant demo
│   ├── sample_portrait.jpg
│   └── sample_group.jpg
└── README.md             # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/amrithlazy-cell/facedetectionapp.git
cd facedetectionapp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*The app will automatically open at `http://localhost:8501` in your browser.*

### 4. (Optional) Run the Native Desktop Webcam App
```bash
python detect_webcam.py
```
- Press `q` to exit.
- Press `b` to toggle privacy blur on faces.
- Press `s` to save a snapshot image.

---

## 🌐 How to Deploy and Host Online

### Option A: Deploy on Streamlit Community Cloud (Recommended)
1. Push your repository to GitHub (`amrithlazy-cell/facedetectionapp`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Select repository: `amrithlazy-cell/facedetectionapp`.
5. Set Main file path to: `app.py`.
6. Click **"Deploy!"** &mdash; Your application will be live with a public URL in 1 minute!

### Option B: Deploy on GitHub Pages
1. Go to your repository on GitHub: `https://github.com/amrithlazy-cell/facedetectionapp`.
2. Click **Settings** &rarr; **Pages** (under Code and automation).
3. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: `main` (or `master`), Folder: `/ (root)`
4. Click **Save**. Within 1–2 minutes, your live web app will be accessible at:
   `https://amrithlazy-cell.github.io/facedetectionapp/`

---

## 🛠️ Technologies Used

- **Python 3.11**
- **OpenCV (Open Source Computer Vision Library)**
- **Streamlit**
- **NumPy & Pillow**
- **MediaPipe (Google AI Edge)**
- **HTML5 Canvas & TailwindCSS**

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
