import streamlit as st
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20

# Initialize session state variables
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "alarm_status" not in st.session_state:
    st.session_state.alarm_status = False
if "alarm_status2" not in st.session_state:
    st.session_state.alarm_status2 = False
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

# Helper functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    return abs(top_mean[1] - low_mean[1])

# Streamlit UI
st.title("Driver Drowsiness Detection System")
st.markdown("Upload a video file to start detecting drowsiness.")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    st.session_state.detection_running = True

if st.session_state.detection_running and uploaded_video is not None:
    # Initialize dlib predictor and detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Read video frames
    vs = cv2.VideoCapture(uploaded_video.name)
    frame_placeholder = st.empty()

    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            st.warning("End of video reached.")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # EAR and Lip Distance calculations
            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # EAR logic
            if ear < EYE_AR_THRESH:
                st.session_state.counter += 1
                if st.session_state.counter >= EYE_AR_CONSEC_FRAMES:
                    if not st.session_state.alarm_status:
                        st.session_state.alarm_status = True
                        st.warning("Drowsiness detected! Please take a break!")
            else:
                st.session_state.counter = 0
                st.session_state.alarm_status = False

            # Yawn detection logic
            if distance > YAWN_THRESH:
                if not st.session_state.alarm_status2:
                    st.session_state.alarm_status2 = True
                    st.warning("Yawning detected! Stay alert!")
            else:
                st.session_state.alarm_status2 = False

        # Display processed video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    vs.release()
    st.session_state.detection_running = False
    st.success("Video processing complete!")
