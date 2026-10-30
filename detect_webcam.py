"""
Real-Time Face Detection Desktop Script
Using OpenCV and Haar Feature-based Cascade Classifiers.

Press 'q' to quit the application.
Press 's' to save a snapshot of the current frame.
Press 'b' to toggle Privacy Blur on detected faces.
"""

import cv2
import sys
import os
import time

def get_cascade_path(filename: str) -> str:
    # 1. Local cascades folder
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades", filename)
    if os.path.exists(local_path):
        return local_path
    # 2. Built-in OpenCV data directory
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cv_path = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(cv_path):
            return cv_path
    return filename

def main():
    print("=" * 60)
    print("  FaceScope AI - Real-Time OpenCV Face Detection")
    print("=" * 60)
    print("Controls:")
    print("  [q] Quit")
    print("  [s] Save Snapshot")
    print("  [b] Toggle Privacy Blur")
    print("-" * 60)

    face_cascade_path = get_cascade_path("haarcascade_frontalface_default.xml")
    eye_cascade_path = get_cascade_path("haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0). Check camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    blur_faces = False
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Calculate FPS
        curr_time = time.time()
        fps = int(1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw detected faces
        for idx, (x, y, w, h) in enumerate(faces):
            if blur_faces:
                # Privacy blur
                sub_face = frame[y:y+h, x:x+w]
                sub_face = cv2.GaussianBlur(sub_face, (99, 99), 30)
                frame[y:y+h, x:x+w] = sub_face
            else:
                # Draw stylish rectangle
                color = (0, 255, 255)  # Yellow-cyan
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Label tag
                label = f"Face #{idx+1}"
                cv2.rectangle(frame, (x, y - 24), (x + 80, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Detect eyes inside face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    eye_center = (ex + ew // 2, ey + eh // 2)
                    radius = int(round((ew + eh) * 0.25))
                    cv2.circle(roi_color, eye_center, radius, (255, 0, 0), 1)

        # Overlay HUD Bar
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (640, 40), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

        status_text = f"Faces: {len(faces)} | FPS: {fps} | Blur: {'ON' if blur_faces else 'OFF'}"
        cv2.putText(frame, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        cv2.imshow("FaceScope AI - Real-Time Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_faces = not blur_faces
        elif key == ord('s'):
            filename = f"face_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[+] Saved snapshot to {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
