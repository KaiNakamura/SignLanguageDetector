import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader



class SoftmaxClassifer(nn.Module): 
   def __init__(self, features: int, num_classes: int) -> None:
      super(SoftmaxClassifer, self).__init__()
      self.linear = nn.Linear(features, num_classes)


   def forward(self, x: float) -> Tensor:
      return self.linear(x)
   

   def preprocess(self, data: np.ndarray) -> tuple:
      y = torch.from_numpy(data[:, 0]).long() # Shape (num_samples,)
      X = torch.from_numpy(data[:, 1:]) # Shape (num_samples, num_features)

      X = X / 255 # Normalize X
      return X, y


   def train (self, data: np.ndarray, num_epochs: int, lr: float, batch_size): 
      X, y = self.preprocess(data)
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
   

   def test (self, data: np.ndarray):
      with torch.no_grad():
         test_X, test_y = self.preprocess(data)
         n = test_y.shape[0]

         logits = self.forward(test_X)
         probabilities = torch.softmax(logits, dim=1)
         predicted_classes = torch.argmax(probabilities, dim=1)
         accuracy = (test_y == predicted_classes).sum().item() / n
         print(f"Test accuracy: {accuracy}")
      


if __name__ == "__main__":
   # Initalize data
   train_data = pd.read_csv('sign_mnist_train.csv').to_numpy(dtype=np.float32)
   test_data = pd.read_csv('sign_mnist_test.csv').to_numpy(dtype=np.float32)

   # Intialize model
   model = SoftmaxClassifer(features=784, num_classes=26)

   # Train
   model.train(train_data, num_epochs=50, lr=1e-1, batch_size=64) 

   # Test 
   model.test(test_data)




   