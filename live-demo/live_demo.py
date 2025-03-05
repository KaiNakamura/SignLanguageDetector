import pickle
import cv2
import mediapipe as mp
import numpy as np
from image_processor import *

model_dict = pickle.load(open("models/model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
)

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_aux, results, x_i, y_i, z_i = process_image(frame_rgb)

    if data_aux and results.multi_hand_landmarks:
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])

        # Draw prediction on screen
        x1 = int(min(x_i) * W) - 10
        y1 = int(min(y_i) * H) - 10

        x2 = int(max(x_i) * W) - 10
        y2 = int(max(y_i) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            prediction[0],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
