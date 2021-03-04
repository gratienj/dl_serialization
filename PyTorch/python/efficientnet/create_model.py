import torch

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b0')
model.eval()

#model.save("model/efficientnet.pt")
smodel = torch.jit.script(model)
model.save("model/traced_efficientnet.pt")
