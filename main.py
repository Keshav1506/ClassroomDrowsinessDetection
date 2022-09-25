from scipy.spatial import distance
from PIL import Image
import asyncio
import os
import numpy
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)
import imutils
from imutils import face_utils
import streamlit as st
from av import VideoFrame as vf

import dlib
import cv2

###################################################################### Operating Code  ######################################################################


#needs work (session states throwing error)

#if 'flag' not in st.session_state:
    #st.session_state.flag = 0


thresh = 0.25
minframes = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]


def framefn(frame):

    img = frame.to_image()
    img.save("temp.jpg")
    img=cv2.imread("temp.jpg")

    frame = imutils.resize(img, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mHull], -1, (0, 255, 0), 1)

        # needs work (session states throwing error)
        # if ear < thresh:
        #     st.session_state.flag+=1
        # else:
        #     st.session_state.flag = 0

        if ear < thresh:                                              #st.session_state.flag > minsecs:
            cv2.putText(frame, "    ALERT!, DROWSINESS DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    Image.fromarray(frame).save("temp.jpg")
    img = Image.open("temp.jpg")
    return vf.from_image(img)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):

	A = distance.euclidean(mouth[2], mouth[10])
	B = distance.euclidean(mouth[4], mouth[8])
	C = distance.euclidean(mouth[0], mouth[6])

	mar = (A + B) / (2.0 * C)

	return mar

#############################################################################################################################################################




###################################################################### User Interface  ######################################################################


#Page Title
st.title("Student Drowsiness Detection")


#Banner Image
st.image("image.jpg")

#Streamer Element
webrtc_streamer(key="opencv-filter",mode=WebRtcMode.SENDRECV,video_frame_callback=framefn,async_processing=True,media_stream_constraints={"video": True, "audio": False},rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})





#############################################################################################################################################################
