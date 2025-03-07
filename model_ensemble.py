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
   
   # Predictions
   knn_train_pred = knn.predict(train_X)
   knn_train_proba = knn.predict_proba(train_X)
   knn_test_pred = knn.predict(test_X)
   knn_test_proba = knn.predict_proba(test_X)

   # Accuracy and Log Loss
   knn_train_accuracy = accuracy_score(train_y, knn_train_pred)
   knn_train_loss = log_loss(train_y, knn_train_proba)
   knn_test_accuracy = accuracy_score(test_y, knn_test_pred)
   knn_test_loss = log_loss(test_y, knn_test_proba)

   print(f"KNN Train Accuracy: {knn_train_accuracy}, Log Loss: {knn_train_loss}")
   print(f"KNN Test Accuracy: {knn_test_accuracy}, Log Loss: {knn_test_loss}")

   # Train Random Forest
   rf = RandomForestClassifier(n_estimators=100, random_state=100)
   rf.fit(train_X, train_y)

   rf_train_pred = rf.predict(train_X)
   rf_train_proba = rf.predict_proba(train_X)
   rf_test_pred = rf.predict(test_X)
   rf_test_proba = rf.predict_proba(test_X)

   rf_train_accuracy = accuracy_score(train_y, rf_train_pred)
   rf_train_loss = log_loss(train_y, rf_train_proba)
   rf_test_accuracy = accuracy_score(test_y, rf_test_pred)
   rf_test_loss = log_loss(test_y, rf_test_proba)

   print(f"Random Forest Train Accuracy: {rf_train_accuracy}, Log Loss: {rf_train_loss}")
   print(f"Random Forest Test Accuracy: {rf_test_accuracy}, Log Loss: {rf_test_loss}")

   # Train Logistic Regression
   clf = OneVsRestClassifier(LogisticRegression(max_iter=500, solver="lbfgs"))
   clf.fit(train_X, train_y)

   pred_train_y = clf.predict(train_X)
   pred_train_proba = clf.predict_proba(train_X)
   pred_test_y = clf.predict(test_X)
   pred_test_proba = clf.predict_proba(test_X)

   train_accuracy = accuracy_score(train_y, pred_train_y)
   train_loss = log_loss(train_y, pred_train_proba)
   test_accuracy = accuracy_score(test_y, pred_test_y)
   test_loss = log_loss(test_y, pred_test_proba)

   print(f"Logistic Regression Train Accuracy: {train_accuracy}, Log Loss: {train_loss}")
   print(f"Logistic Regression Test Accuracy: {test_accuracy}, Log Loss: {test_loss}")