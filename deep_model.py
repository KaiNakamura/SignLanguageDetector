import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
import optuna 


class ConvolutionalNeuralNetwork(nn.Module):
   def __init__(self, num_classes: int, num_dense_nodes: int):
      super(ConvolutionalNeuralNetwork, self).__init__()
      self.conv_layer = nn.Sequential(
         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),

         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),

         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2) 
      )
      self.dense_layer = nn.Sequential(
         nn.Linear(64 * 4 * 4, num_dense_nodes),
         nn.Dropout(0.5),
         nn.Linear(num_dense_nodes, num_classes)
      )


   def forward(self, X: np.ndarray): 
      X = self.conv_layer(X)
      return self.dense_layer(X)
   

   def train_model(self, train_X: np.ndarray, train_y: np.ndarray, num_epochs: int, lr: float, batch_size: int):
      dataset = TensorDataset(train_X, train_y)
      loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

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
   
   
   def test_model (self, test_X: np.ndarray, test_y: np.ndarray) -> float:
      with torch.no_grad():
         n = test_y.shape[0]

         logits = self.forward(test_X)
         probabilities = torch.softmax(logits, dim=1)
         predicted_classes = torch.argmax(probabilities, dim=1)

         accuracy = (test_y == predicted_classes).sum().item() / n
         print(f"Test accuracy: {accuracy}")
         return accuracy


def preprocess(data: pd.DataFrame, device: torch.device) -> tuple:

   # In order to fix this subtract all labels by 1 if label is > 9
   data['label'] = data['label'].apply(lambda x: x - 1 if x > 9 else x)

   # Convert to np
   data = data.to_numpy(dtype=np.float32)
   
   y = torch.from_numpy(data[:, 0]).long().to(device) # Shape (num_samples,)
   X = torch.from_numpy(data[:, 1:]).to(device) # Shape (num_samples, num_features)

   X = X / 255 # Normalize X
   return X, y

if __name__ == "__main__":

   pass