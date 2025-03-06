import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, log_loss

def preprocess(data: pd.DataFrame) -> tuple:

   # In order to fix this subtract all labels by 1 if label is > 9
   data["label"] = data["label"].apply(lambda x: x - 1 if x > 9 else x)

   # Convert to np
   data = data.to_numpy(dtype=np.float32)

   y = torch.from_numpy(data[:, 0]).long() # Shape (num_samples,)
   X = torch.from_numpy(data[:, 1:]) # Shape (num_samples, num_features)

   X = X / 255  # Normalize X
   return X, y

if __name__ == "__main__":

   # Initalize data
   train_data = pd.read_csv("data/sign_mnist_train.csv")
   test_data = pd.read_csv("data/sign_mnist_test.csv")

   train_X, train_y = preprocess(train_data)
   test_X, test_y = preprocess(test_data)

   # Train KNN
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(train_X, train_y)
   knn_pred = knn.predict(test_X)
   knn_proba = knn.predict_proba(test_X)  
   knn_accuracy = accuracy_score(test_y, knn_pred)
   knn_loss = log_loss(test_y, knn_proba)
   print(f"KNN Test Accuracy: {knn_accuracy}, Log Loss: {knn_loss}")

   # Train Random Forest
   rf = RandomForestClassifier(n_estimators=100, random_state=100)
   rf.fit(train_X, train_y)
   rf_pred = rf.predict(test_X)
   rf_proba = rf.predict_proba(test_X)
   rf_accuracy = accuracy_score(test_y, rf_pred)
   rf_loss = log_loss(test_y, rf_proba)
   print(f"Random Forest Test Accuracy: {rf_accuracy}, Log Loss: {rf_loss}")

   # Train logistic regression
   clf = OneVsRestClassifier(LogisticRegression(max_iter=500, solver="lbfgs"))
   clf.fit(train_X, train_y)
   pred_y = clf.predict(test_X)
   pred_proba = clf.predict_proba(test_X)
   accuracy = accuracy_score(test_y, pred_y)
   loss = log_loss(test_y, pred_proba)
   print(f"Logistic Regression Test Accuracy: {accuracy}, Log Loss: {loss}")