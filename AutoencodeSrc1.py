import torch
import torch.nn as nn
class AutoencoderX1(nn.Module):
    
    def __init__(self):
        super().__init__()
        #create encoder layers 
        self.encoder=nn.Sequential(
            nn.Linear(in_features=60,out_features=40),
            nn.Linear(in_features=40,out_features=20)
        )
        #create decoder layers 
        self.decoder=nn.Sequential(
            nn.Linear(in_features=20,out_features=40),
            nn.Linear(in_features=40,out_features=60)
        )
        
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded 
       
