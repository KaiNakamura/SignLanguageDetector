import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import optuna


class SoftmaxClassifer(nn.Module):
   def __init__(self, features: int, num_classes: int) -> None:
      super(SoftmaxClassifer, self).__init__()
      self.linear = nn.Linear(features, num_classes)

   def forward(self, x: Tensor) -> Tensor:
      return self.linear(x)

   def train_model(self, train_X: Tensor, train_y: Tensor, num_epochs: int, lr: float, batch_size: int):
      dataset = TensorDataset(train_X, train_y)
      loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

      loss_fn = nn.CrossEntropyLoss()  # CE loss runs softmax function
      optimizer = optim.SGD(self.parameters(), lr=lr)

      for epoch in range(1, num_epochs + 1):
         for batch_x, batch_y in loader:
               optimizer.zero_grad()
               yhat = self.forward(batch_x)
               loss = loss_fn(yhat, batch_y)
               loss.backward()
               optimizer.step()

         if epoch % 10 == 0:
               print(f"Epoch {epoch} / {num_epochs}: {loss.item()}")

   def test_model(self, test_X: Tensor, test_y: Tensor, which: str) -> tuple:
      with torch.no_grad():
         n = test_y.shape[0]
         loss_fn = nn.CrossEntropyLoss()  # CE loss runs softmax function

         logits = self.forward(test_X)
         probabilities = torch.softmax(logits, dim=1)
         loss = loss_fn(logits, test_y)
         predicted_classes = torch.argmax(probabilities, dim=1)

         accuracy = (test_y == predicted_classes).sum().item() / n
         print(f"{which} accuracy: {accuracy}")
         print(f"{which} loss: {loss}")
         return accuracy, loss


def preprocess(data: pd.DataFrame, device: torch.device) -> tuple:

   # In order to fix this subtract all labels by 1 if label is > 9
   data["label"] = data["label"].apply(lambda x: x - 1 if x > 9 else x)

   # Convert to np
   data = data.to_numpy(dtype=np.float32)

   y = torch.from_numpy(data[:, 0]).long().to(device)  # Shape (num_samples,)
   X = torch.from_numpy(data[:, 1:]).to(device)  # Shape (num_samples, num_features)

   X = X / 255  # Normalize X
   return X, y


def Objective(trial: optuna.Trial) -> float:
   num_epochs = trial.suggest_categorical("num_epochs", [10, 20, 50, 100])
   lr = trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
   batch_size = trial.suggest_categorical("batch_size", [10, 32, 64, 128, 256])

   # Initalize data
   train_data = pd.read_csv("data/sign_mnist_train.csv")
   test_data = pd.read_csv("data/sign_mnist_test.csv")

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   train_X, train_y = preprocess(train_data, device)
   test_X, test_y = preprocess(test_data, device)

   # Intialize model
   model = SoftmaxClassifer(features=784, num_classes=24).to(device)

   # Train
   model.train_model(train_X=train_X, train_y=train_y, num_epochs=num_epochs, lr=lr, batch_size=batch_size)

   # Test
   accuracy, loss = model.test_model(test_X=test_X, test_y=test_y, which="Test")

   return accuracy


if __name__ == "__main__":

#  # Hyperparameter tuning
#  study = optuna.create_study(direction="maximize")  # Maximize accuracy
#  study.optimize(Objective, n_trials=50)

#  # Print best hyperparameters
#  print("Best hyperparameters:", study.best_params)

   # Best hyperparameters: {'num_epochs': 100, 'lr': 0.01, 'batch_size': 10}

   # # Initalize data
   train_data = pd.read_csv("data/sign_mnist_train.csv")
   test_data = pd.read_csv("data/sign_mnist_test.csv")

   device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
   )  # Select use of GPU/CPU

   train_X, train_y = preprocess(train_data, device)
   test_X, test_y = preprocess(test_data, device)

   # Intialize model
   model = SoftmaxClassifer(features=784, num_classes=24).to(device)

   # Train
   model.train_model(train_X=train_X, train_y=train_y, num_epochs=100, lr=0.01, batch_size=10)

   # Training Metrics
   accuracy, loss = model.test_model(test_X=train_X, test_y=train_y, which="Train")

   # Test
   accuracy, loss = model.test_model(test_X=test_X, test_y=test_y, which="Test")
