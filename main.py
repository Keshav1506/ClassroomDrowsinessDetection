from scipy.spatial import distance
from PIL import Image
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

thresh = 0.25
ear=0.0
flag=0
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]



def framefn(frame):
    global ear,flag
    img = frame.to_image()
    img.save("test.jpg")
    img=cv2.imread("test.jpg")

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
        ears = (leftEAR + rightEAR) / 2.0
        ear= ears
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mHull], -1, (0, 255, 0), 1)

        if ear <= 0.5:
            flag += 1


    Image.fromarray(frame).save("test1.jpg")
    img = Image.open("test1.jpg")
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

st.title("Student Drowsiness Detection")
st.image("image.jfif")




status = st.empty()
infnc=st.empty()
framenumber= st.empty()


frame_check = 10

webrtc_streamer(key="opencv-filter",mode=WebRtcMode.SENDRECV,video_frame_callback=framefn,async_processing=False,media_stream_constraints={"video": True, "audio": False})

infnc.text(ear)
framenumber.text(flag)
if flag >= frame_check:
    status.markdown("<h1 style='text-align: center; color: red;'>Drowsy!</h1>", unsafe_allow_html=True)


else:
    status.markdown("<h1 style='text-align: center; color: green;'>Active</h1>", unsafe_allow_html=True)
    flag = 0




