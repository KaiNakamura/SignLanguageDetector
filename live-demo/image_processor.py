import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
)


def process_image(img_rgb, display_landmarks=False):
    data_aux = []
    x_i = []
    y_i = []
    z_i = []

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_i.append(x)
                y_i.append(y)
                z_i.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append(x - min(x_i))
                data_aux.append(y - min(y_i))
                data_aux.append(z - min(z_i))

            # Optionally, display the image with landmarks
            if display_landmarks:
                draw_landmarks(img_rgb, hand_landmarks)
                cv2.imshow("Hand Landmarks", img_rgb)
                cv2.waitKey(0)

    return data_aux, results, x_i, y_i, z_i


def draw_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
