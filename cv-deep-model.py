import cv2
import numpy as np
import torch
import mediapipe as mp
from torchvision import transforms
from PIL import Image
from deep_model import ConvolutionalNeuralNetwork

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Define transformations (ensure grayscale, normalize, and convert to tensor)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

# ASL letters corresponding to the 24 classes (excluding J and Z)
asl_classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalNeuralNetwork(num_classes=24, num_dense_nodes=64)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model.to(device)  # Move model to device
model.eval()

# Hand detection
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand bounding box
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Expand bounding box slightly
                box_size = max(x_max - x_min, y_max - y_min)
                x_min = max(x_min - box_size // 4, 0)
                y_min = max(y_min - box_size // 4, 0)
                x_max = min(x_max + box_size // 4, w)
                y_max = min(y_max + box_size // 4, h)

                # Extract hand region
                hand_region = frame[y_min:y_max, x_min:x_max]

                # Resize to 56x56 first, then downsample to 28x28
                resized_hand = cv2.resize(hand_region, (28, 28))

                # Convert to PIL Image for PyTorch processing
                resized_hand_pil = Image.fromarray(resized_hand)

                # Apply transformation
                tensor_hand = transform(resized_hand_pil).unsqueeze(0).to(device)  # Add batch dimension

                # Predict
                with torch.no_grad():
                    output = model(tensor_hand)
                    predicted_class = torch.argmax(output, dim=1).item()

                    predicted_letter = asl_classes.get(predicted_class, "?")  # Get letter, default to '?'


                # Display prediction
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted: {predicted_letter}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show frame
        cv2.imshow("Hand Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
