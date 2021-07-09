import torch
import os
import sys
from models import Sincnetmodelv2

device = 'cpu'

model = Sincnetmodelv2().to(device)

model.load_state_dict(torch.load("./models/model_logical_wave_100_128_0.0001_20000/final.pth", map_location=torch.device('cpu'))['state_dict'])

torch.set_printoptions(profile="full")

f = open("./outputs/model-200713.txt", "w")

#for name, param in model.named_parameters():
#      if param.requires_grad:
#              print("Name: {},\n Param:\n{}\n".format(name, param.data), file=f)

#print(model, file=f)

low_hz = model.state_dict()["tConv2.low_hz_"]
band_hz = model.state_dict()["tConv2.band_hz_"]
weight = model.state_dict()["se1.fc.2.weight"]

print(low_hz.size(), "\n", band_hz.size(), "\n", weight.size(), file=f)

print("\n======\n", file=f)

torch.set_printoptions(profile="full")
print(low_hz, "\n", band_hz, "\n", weight, file=f)
torch.set_printoptions(profile="default")


torch.set_printoptions(profile="default")
f.close()


#model_save_path = "../models/model_logical_wave_3_128_0.0001_100"

#model = torch.load(os.path.join("models", model_save_path, "final.pth"), map_location=torch.device('cpu'))

#for name, param in model.named_parameters():
#	if param.requires_grad:
#		print("Name:{}    Param:{}\n".format(name, param.data))

#print("========\n\n")

#summary(model, (3, 224, 224))



#params = model.state_dict()

#print(params["tConv1.weight"])

#print("======\n")

#print(params["tConv1.bias"])
