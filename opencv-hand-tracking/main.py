import numpy as np
import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
WIN_HEIGHT = 600
WIN_WIDTH = 600

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
cap.set(cv.CAP_PROP_FRAME_HEIGHT, WIN_HEIGHT)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIN_WIDTH)

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
    
while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    
    if success:        
        RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        
        if result.multi_hand_landmarks:
            hand_landmarks_set = result.multi_hand_landmarks
            for hand_landmarks in hand_landmarks_set:
                # For sending the image to the model, we will probably need to remove the landmarks drawing
                # since the model will only need the cropped hand image
                print(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
                # draw a square box around each hand
                x_min, y_min = 1000, 1000
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y
                
                # Calculate the center and size of the square
                box_size = max(x_max - x_min, y_max - y_min)
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                
                # Calculate new square coordinates
                x_min = center_x - box_size // 2
                x_max = center_x + box_size // 2
                y_min = center_y - box_size // 2
                y_max = center_y + box_size // 2
                
                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # TODO: crop the hand image, resize it to the model input size and send it to the model
                # cropped_hand = frame[y_min:y_max, x_min:x_max]
                # resized_hand = cv.resize(cropped_hand, (28, 28))
                
        cv.imshow("capture image", frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()