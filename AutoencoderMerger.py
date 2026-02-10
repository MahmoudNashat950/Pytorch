import torch
import torch.nn as nn
from AutoencodeSrc1  import AutoencoderX1
from AutoencodeSrc2 import AutoencoderX2
from  AutoencodeSrc  import AutoencoderX

class AutoencoderMerger(nn.Module):
  
   def __init__(self):
      super().__init__()
      self.ae_x1=AutoencoderX1()
      self.ae_x2=AutoencoderX2()
      self.ae_x=AutoencoderX()


   def forward (self,x1,x2):
      x1=self.ae_x1(x1)
      x2=self.ae_x2(x2)
      x=torch.hstack([x1,x2])
      #for skip connection (just add + operator)
      x+=self.ae_x(x)

      return x1,x2,x