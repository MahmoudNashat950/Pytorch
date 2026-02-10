import torch
import torch.nn as nn

class AutoencoderX(nn.Module):
  
   def __init__(self):
      super().__init__()
      #create encoder layers
      self.encoder=nn.Sequential(
         nn.Linear(in_features=100,out_features=75),
         nn.Linear(in_features=75,out_features=25)

      )
      #create decoder layers
      self.decoder=nn.Sequential(
         nn.Linear(in_features=25,out_features=75),
         nn.Linear(in_features=75,out_features=100)

      )


   def forward (self,x):
      encoded=self.encoder(x)  
      decoded=self.decoder(encoded)
      return decoded