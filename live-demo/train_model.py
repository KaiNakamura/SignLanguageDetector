import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dir = "data/landmark_data"
data = []
labels = []

# Load all pickle files from the data directory
for filename in os.listdir(data_dir):
    if filename.endswith(".pickle"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
            data.extend(data_dict["data"])
            labels.extend(data_dict["labels"])

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"Testing Accuracy: {score * 100:.2f}%")

f = open("models/model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
