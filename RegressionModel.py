import torch
import torch.nn as nn
from AutoencoderMerger import AutoencoderMerger
class RegressionModel (nn.Module):
  
   def __init__(self):
      super().__init__()
      #autoencoder to process inputs
      self.autoenoder=AutoencoderMerger()
      #create the layers
      self.model=nn.Sequential(
         nn.Linear(in_features=100,out_features=50),
         nn.ReLU(),
         nn.Linear(50,25),
         nn.ReLU(),
         nn.Linear(25,1)
      )
      


   def forward (self,x1,x2):
      x1,x2,x=self.autoenoder(x1,x2)
      predication=self.model(x)
      return x1 ,x2 ,x ,predication  # to apply losses
     