


import streamlit as st
import imutils
import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import time
from scipy.spatial import distance as dist


# Initialize variables
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0  # Declare COUNTER here


# Alarm function
def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        playsound.playsound(path)
    if alarm_status2:
        saying = True
        playsound.playsound(path)
        saying = False

# Eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Final EAR function
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

# Lip distance function
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
st.markdown("Click the button below to start detecting drowsiness.")

# Start button for detection
if st.button("Start Detection"):
    st.text("Loading the predictor and detector...")

    # Initialize Haar Cascade and dlib predictor
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()
    alarm_status = False
    alarm_status2 = False

    # Process video frames
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)  # Resize the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Eye and Lip detection
            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # EAR threshold logic
            #global COUNTER  # Declare COUNTER as global here
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        Thread(target=sound_alarm, args=("Alert.wav",)).start()
            else:
                COUNTER = 0
                alarm_status = False

            # Yawn detection logic
            if distance > YAWN_THRESH:
                if not alarm_status2:
                    alarm_status2 = True
                    Thread(target=sound_alarm, args=("music.wav",)).start()
            else:
                alarm_status2 = False

        # Display only the live video feed, no text alerts
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()
    st.text("Detection stopped.")
