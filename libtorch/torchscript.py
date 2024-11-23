import torch
import torchvision

model = torchvision.models.resnet18()                           # --- (1)
sample_input = torch.rand(1, 3, 224, 224)                       # --- (2)
traced_script_module = torch.jit.trace(model, sample_input)     # --- (3)
traced_script_module.save("/Users/jongbeomkim/Downloads/traced_script_model.pt")	 # --- (1)
