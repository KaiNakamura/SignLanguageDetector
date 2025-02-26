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


   def forward(self, x: float) -> Tensor:
      return self.linear(x)
   

   def preprocess(self, data: np.ndarray, device: torch.device) -> tuple:
      y = torch.from_numpy(data[:, 0]).long().to(device) # Shape (num_samples,)
      X = torch.from_numpy(data[:, 1:]).to(device) # Shape (num_samples, num_features)

      X = X / 255 # Normalize X
      return X, y


   def train (self, data: np.ndarray, num_epochs: int, lr: float, batch_size, device: torch.device): 
      X, y = self.preprocess(data, device)
      dataset = TensorDataset(X, y)
      loader = DataLoader(dataset=dataset, batch_size=batch_size)

      loss_fn = nn.CrossEntropyLoss() # CE loss runs softmax function
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
   

   def test (self, data: np.ndarray, device: torch.device):
      with torch.no_grad():
         test_X, test_y = self.preprocess(data, device)
         n = test_y.shape[0]

         logits = self.forward(test_X)
         probabilities = torch.softmax(logits, dim=1)
         predicted_classes = torch.argmax(probabilities, dim=1)
         accuracy = (test_y == predicted_classes).sum().item() / n
         print(f"Test accuracy: {accuracy}")
         return accuracy
      

def Objective(trial: optuna.Trial) -> float:
   num_epochs = trial.suggest_categorical("num_epochs", [10, 20, 50, 100])
   lr = trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
   batch_size = trial.suggest_categorical('batch_size', [10, 32, 64, 128, 256])

   # Initalize data
   train_data = pd.read_csv('sign_mnist_train.csv').to_numpy(dtype=np.float32)
   test_data = pd.read_csv('sign_mnist_test.csv').to_numpy(dtype=np.float32)

   # Intialize model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = SoftmaxClassifer(features=784, num_classes=26).to(device)

   # Train
   model.train(data=train_data, num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=device) 

   # Test 
   accuracy = model.test(data=test_data, device=device) 



if __name__ == "__main__":

   # Hyperparameter tuning
   study = optuna.create_study(direction="maximize")  # Maximize accuracy
   study.optimize(Objective, n_trials=30)  # Run 30 trials

   # Print best hyperparameters
   print("Best hyperparameters:", study.best_params)


   # Initalize data
   # train_data = pd.read_csv('sign_mnist_train.csv').to_numpy(dtype=np.float32)
   # test_data = pd.read_csv('sign_mnist_test.csv').to_numpy(dtype=np.float32)

   # Intialize model
   # device = torch.device("cude" if torch.cuda.is_available() else "cpu")
   # model = SoftmaxClassifer(features=784, num_classes=26).to(device)

   # Train
   # model.train(train_data, num_epochs=50, lr=1e-1, batch_size=64) 

   # # Test 
   # model.test(test_data)
  