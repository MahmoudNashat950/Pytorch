import torch
import torch.nn as nn

import RegressionModel
import AutoencodeSrc1
import AutoencodeSrc2
import AutoencodeSrc
import drop_features

drop_prob=0.9
critertion=nn.MSELoss()
regressorModel=RegressionModel()

#data
x1=torch.rand(size=[32,60])
x2=torch.rand(size=[32,40])
x=torch.hstack([x1,x2])
gt=torch.randn(32,1)

#output from endcoders
x1_noise=drop_features(x1,drop_prob)
x2_nosise=drop_features(x2,drop_prob)

x1_rec,x2_rec,x_rec,prediction=regressorModel(x1_noise,x2_nosise)
#loss
loss1=critertion(x1,x1_rec)
loss2=critertion(x2,x2_rec)
loss3=critertion(x,x_rec)
loss4=critertion(gt,prediction)

loss=(loss1+loss2+loss3+loss4)/4  #  الحمدلله
