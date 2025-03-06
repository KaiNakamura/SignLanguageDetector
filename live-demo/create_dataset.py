import os
import pickle
import cv2
import random
from image_processor import process_image
from datetime import datetime

DATA_DIR = "data/custom_recording_RUV_3000"

data = []
labels = []

# Process each image in the dataset
print("Begining processing images...")
for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        # Get rgb image
        img_full_path = os.path.join(DATA_DIR, dir, img_path)
        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Randomly flip the image
        # Don't need to worry about left vs. right hand
        if random.random() < 0.5:
            img_rgb = cv2.flip(img_rgb, 1)

        # Process the image
        data_aux, _, _, _, _ = process_image(img_rgb)

        if data_aux:
            data.append(data_aux)
            labels.append(dir)

    print(f"Finished processing {dir}")
print("Finished processing all images!")

# Save the data and labels to a pickle file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/landmark_data/{timestamp}.pickle"
os.makedirs(os.path.dirname(filename), exist_ok=True)
f = open(filename, "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
