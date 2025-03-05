import os
import cv2


def draw_text(
    frame, text, position=(10, 30), font_scale=0.8, color=(0, 0, 0), thickness=1
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# No J or Z since they require gestures to properly sign
classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]
dataset_size = 500

DATA_DIR = f"data/custom_recording_{dataset_size}"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        draw_text(
            frame,
            f"Class: {class_name}, Press any key to start...",
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) != -1:
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        draw_text(frame, f"Recording: {class_name}", color=(255, 0, 0))
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, "{}.jpg".format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
