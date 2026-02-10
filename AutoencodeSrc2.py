import torch
import torch.nn as nn

class AutoencoderX2(nn.Module):
  
   def __init__(self):
      super(AutoencoderX2,self).__init__()
      #create encoder layers
      self.encoder=nn.Sequential(
         nn.Linear(in_features=40,out_features=15)
      )
      #create decoder layers
      self.decoder=nn.Sequential(
         nn.Linear(in_features=15,out_features=40)
      )


   def forward (self,x):
      encoded=self.encoder(x)  
      decoded=self.decoder(encoded)
      return decoded
