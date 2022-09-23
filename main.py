from scipy.spatial import distance
import imutils
from imutils import face_utils
import streamlit as st

import dlib
import cv2




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




placeholder = st.empty()
status = st.empty()


frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(MoStart, MoEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


if st.button("Start proctoring"):

    cap = cv2.VideoCapture(0)


    flag = 0
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:

            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            mouth = shape[MoStart:MoEnd]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouthEAR = mouth_aspect_ratio(mouth)
            thresh=0.25 #thresh
            ear = (leftEAR + rightEAR ) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            placeholder.image(frame)




            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    status.markdown("<h1 style='text-align: center; color: red;'>Drowsy!</h1>",unsafe_allow_html=True)


            else:
                status.markdown("<h1 style='text-align: center; color: green;'>Active</h1>",unsafe_allow_html=True)
                flag = 0

if st.button("Stop proctoring"):
    st.stop()






